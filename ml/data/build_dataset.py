"""
ml/data/build_dataset.py — Fusiona fuentes crudas y construye el dataset de entrenamiento.

Entradas:
    data/raw/esios/     → demanda, generación, intercambios, reserva (10 min / hora)
    data/raw/aemet/     → temperatura, viento, radiación (hora)

Salida:
    data/processed/dataset_training.parquet
        Índice: timestamp UTC (horario)

Features generadas:
    · Demanda, generación por tecnología, intercambios, reserva (de ESIOS)
    · Temperatura media nacional, velocidad viento media, radiación media (AEMET)
    · Toda la capa derivada la produce ml.features.build_features (fuente única,
      paridad train/serve garantizada): proxies de estabilidad, inercia (NSP),
      estrés de interconexión por enlace, deltas, volatilidad, rolling composite,
      z-scores 7d, lags y features temporales. Ver ml/features.py::FEATURE_COLS.

Uso:
    python -m ml.data.build_dataset --esios data/raw/esios --aemet data/raw/aemet
                                    --out data/processed
    python -m ml.data.build_dataset            # usa rutas por defecto
    python -m ml.data.build_dataset --refresh  # re-deriva features in-place sobre
                                               # el parquet existente (sin re-descarga)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from ml.features import FEATURE_COLS, build_features

log = logging.getLogger("build_dataset")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)


# ---------------------------------------------------------------------------
# Carga de fuentes crudas
# ---------------------------------------------------------------------------

def _load_parquet_dir(directory: Path, glob: str = "*.parquet") -> pd.DataFrame:
    """Carga y concatena todos los parquets de un directorio."""
    files = sorted(directory.glob(glob))
    if not files:
        log.warning("Sin archivos en %s con patrón '%s'", directory, glob)
        return pd.DataFrame()
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames)
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df.index = pd.DatetimeIndex(df.index, name="timestamp")
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df.sort_index()


def load_esios(esios_dir: Path) -> pd.DataFrame:
    """Carga datos ESIOS y los remuestrea a granularidad horaria."""
    log.info("Cargando datos ESIOS desde %s…", esios_dir)
    parts: dict[str, pd.DataFrame] = {}

    for category in ("demand", "generation", "interchange", "reserve"):
        cat_dir = esios_dir / category
        if cat_dir.exists():
            df = _load_parquet_dir(cat_dir)
            if not df.empty:
                parts[category] = df

    if not parts:
        log.warning("Sin datos ESIOS disponibles")
        return pd.DataFrame()

    # Remuestreo a hora (mean) para alinear 10-min con hora
    resampled = {k: v.resample("h").mean() for k, v in parts.items()}
    merged = pd.concat(resampled.values(), axis=1)

    # Corrección de unidad: indicador ESIOS 1293 (demanda real) devuelve valores
    # en unidades de MW×2 para granularidad de 10 min. Tras resampleado (media),
    # el factor se preserva → dividir por 2 para obtener MW reales.
    if "demand_real_mw" in merged.columns:
        merged["demand_real_mw"] = merged["demand_real_mw"] / 2.0
        log.info(
            "demand_real_mw corregida (÷2): rango %.0f–%.0f MW",
            merged["demand_real_mw"].min(),
            merged["demand_real_mw"].max(),
        )

    log.info("ESIOS cargado: %d filas, %d columnas", len(merged), merged.shape[1])
    return merged


def load_aemet(aemet_dir: Path) -> pd.DataFrame:
    """Carga observaciones AEMET y calcula medias nacionales por hora."""
    log.info("Cargando datos AEMET desde %s…", aemet_dir)
    df = _load_parquet_dir(aemet_dir, "observations_*.parquet")
    if df.empty:
        return pd.DataFrame()

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    # Media nacional diaria (promedio de todas las estaciones para cada día)
    daily = df[numeric_cols].resample("D").mean()
    # Expandir a índice horario: propagar el valor diario a las 24h del día
    hourly_idx = pd.date_range(daily.index.min(), daily.index.max() + pd.Timedelta("23h"), freq="h", tz="UTC")
    hourly = daily.reindex(hourly_idx).ffill().bfill()
    hourly.columns = [f"aemet_{c}" for c in hourly.columns]
    log.info("AEMET cargado: %d filas, %d columnas", len(hourly), hourly.shape[1])
    return hourly


# ---------------------------------------------------------------------------
# Feature engineering — delegada por completo en ml.features (fuente única)
# ---------------------------------------------------------------------------

def add_canonical_reserve(df: pd.DataFrame) -> pd.DataFrame:
    """Construye reserve_mw canónico desde las columnas ESIOS de reserva secundaria.

    ml.features espera una única columna reserve_mw; ESIOS publica energía
    secundaria a subir/bajar por separado (MWh horarios ≈ MW medios).
    """
    reserve_cols = [c for c in df.columns if "secondary_reserve" in c]
    df["reserve_mw"] = df[reserve_cols].abs().sum(axis=1) if reserve_cols else 0.0
    return df


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """reserve_mw canónico + capa completa de ml.features, con verificación."""
    df = add_canonical_reserve(df)
    df = build_features(df)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"build_features no produjo columnas esperadas: {missing}")
    return df


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def build_dataset(
    esios_dir: Path,
    aemet_dir: Path,
    out_dir: Path,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Path:
    esios = load_esios(esios_dir)
    aemet = load_aemet(aemet_dir)

    available = [df for df in (esios, aemet) if not df.empty]
    if not available:
        raise RuntimeError(
            "No hay datos disponibles. Ejecutar primero download_esios.py "
            "y download_aemet.py."
        )

    log.info("Fusionando fuentes en índice horario UTC…")
    dataset = pd.concat(available, axis=1)

    if start:
        dataset = dataset[dataset.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        dataset = dataset[dataset.index <= pd.Timestamp(end, tz="UTC")]

    # Remuestreo final al índice horario completo (rellena huecos con NaN)
    hourly_idx = pd.date_range(
        start=dataset.index.min(),
        end=dataset.index.max(),
        freq="h",
        tz="UTC",
    )
    dataset = dataset.reindex(hourly_idx)
    dataset.index.name = "timestamp"

    log.info("Derivando features (ml.features)…")
    dataset = derive_features(dataset)

    # Estadísticas de calidad
    missing_pct = dataset.isna().mean() * 100
    high_missing = missing_pct[missing_pct > 20]
    if not high_missing.empty:
        log.warning(
            "Columnas con >20%% valores ausentes:\n%s",
            high_missing.to_string()
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dataset_training.parquet"
    dataset.to_parquet(out_path, engine="pyarrow", compression="snappy")

    log.info(
        "Dataset guardado: %s | %d filas × %d columnas | %.1f MB",
        out_path,
        len(dataset),
        dataset.shape[1],
        out_path.stat().st_size / 1e6,
    )
    return out_path


def refresh_features(parquet_path: Path) -> Path:
    """Re-deriva la capa de features in-place sobre el parquet existente.

    Las columnas raw (demanda, generación por tecnología, intercambios, reserva
    secundaria, AEMET) se conservan; toda la capa derivada se recalcula con
    ml.features y sobrescribe la anterior. No re-descarga nada.
    """
    log.info("Refrescando features in-place: %s", parquet_path)
    dataset = pd.read_parquet(parquet_path)
    dataset = derive_features(dataset)
    dataset.to_parquet(parquet_path, engine="pyarrow", compression="snappy")
    log.info(
        "Dataset refrescado: %d filas × %d columnas | %.1f MB",
        len(dataset), dataset.shape[1], parquet_path.stat().st_size / 1e6,
    )
    return parquet_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construye dataset de entrenamiento")
    parser.add_argument("--esios", default="data/raw/esios")
    parser.add_argument("--aemet", default="data/raw/aemet")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--start", default=None, help="Filtro inicio (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="Filtro fin (YYYY-MM-DD)")
    parser.add_argument("--refresh", action="store_true",
                        help="Re-derivar features in-place sobre el parquet existente")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.refresh:
        refresh_features(Path(args.out) / "dataset_training.parquet")
    else:
        build_dataset(
            esios_dir=Path(args.esios),
            aemet_dir=Path(args.aemet),
            out_dir=Path(args.out),
            start=args.start,
            end=args.end,
        )
