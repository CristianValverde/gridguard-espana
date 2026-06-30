"""
ml/data/build_dataset.py — Fusiona fuentes crudas y construye el dataset de entrenamiento.

Entradas:
    data/raw/esios/     → demanda, generación, intercambios, reserva (10 min / hora)
    data/raw/aemet/     → temperatura, viento, radiación (hora)
    data/raw/entsoe/    → flujos transfronterizos, generación vecinos (hora)

Salida:
    data/processed/dataset_training.parquet
        Índice: timestamp UTC (horario)
        ~525.000 filas para 2014→2024 · ~50 features

Features generadas:
    · Demanda, generación por tecnología, intercambios, reserva (de ESIOS)
    · Temperatura media nacional, velocidad viento media, radiación media (AEMET)
    · Flujos transfronterizos EU (ENTSO-E)
    · Proxies de estabilidad calculados (ver ARCHITECTURE.md §5, Capa 1):
        - balance_mw: generación_total − demanda
        - reserve_activated_pct: reserva activada / reserva disponible × 100
        - interchange_delta_mw: |variación en flujo neto externo| (rolling 1h)
        - renewables_pct: (solar + eólica) / demanda × 100
        - composite_pressure: combinación normalizada de los 4 proxies
    · Features temporales:
        - hour_sin, hour_cos (encoding circular)
        - dow_sin, dow_cos (día de semana circular)
        - month_sin, month_cos (mes circular)
        - is_weekend, is_holiday_es

Uso:
    python -m ml.data.build_dataset --esios data/raw/esios --aemet data/raw/aemet
                                    --entsoe data/raw/entsoe --out data/processed
    python -m ml.data.build_dataset  # usa rutas por defecto
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from tools.stability_proxy_calculator import compute_composite_pressure_score

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


def load_entsoe(entsoe_dir: Path) -> pd.DataFrame:
    """Carga flujos transfronterizos ENTSO-E."""
    log.info("Cargando datos ENTSO-E desde %s…", entsoe_dir)
    cb_dir = entsoe_dir / "crossborder"
    df = _load_parquet_dir(cb_dir) if cb_dir.exists() else pd.DataFrame()
    if not df.empty:
        df = df.resample("h").mean()
        log.info("ENTSO-E cargado: %d filas, %d columnas", len(df), df.shape[1])
    return df


# ---------------------------------------------------------------------------
# Feature engineering — Proxies de estabilidad (ARCHITECTURE.md §5 Capa 1)
# ---------------------------------------------------------------------------

def compute_stability_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula los 4 proxies de estabilidad de frecuencia.

    Sustituyen a los datos de frecuencia real (no disponibles públicamente vía API).
    """
    # 1. Balance oferta/demanda (MW)
    gen_cols = [c for c in df.columns if c.startswith(("nuclear", "hydro", "wind",
                "solar", "gas_cc", "coal", "cogeneration"))]
    if gen_cols and "demand_real_mw" in df.columns:
        df["balance_mw"] = df[gen_cols].sum(axis=1) - df["demand_real_mw"]
    else:
        df["balance_mw"] = np.nan

    # 2. % Reserva de regulación activada
    # ESIOS: secondary_reserve_up_mwh + secondary_reserve_dn_mwh (sumamos valor absoluto)
    reserve_cols = [c for c in df.columns if "secondary_reserve" in c]
    if reserve_cols and "demand_real_mw" in df.columns:
        reserve_total = df[reserve_cols].abs().sum(axis=1)
        max_reserve = df["demand_real_mw"] * 0.015  # ~1.5% de la demanda → reserva máxima típica
        df["reserve_activated_pct"] = (
            reserve_total / max_reserve.replace(0, np.nan) * 100
        ).clip(0, 100)
    else:
        df["reserve_activated_pct"] = np.nan

    # 3. Variación brusca en flujos netos de intercambio (rolling 1h)
    # ENTSO-E: columnas con prefijo "flow_"; ESIOS: columnas es_fr_mw, es_pt_mw, es_ma_mw
    flow_cols = [
        c for c in df.columns
        if ("flow_" in c and c.endswith("_mw"))
        or c in ("es_fr_mw", "es_pt_mw", "es_ma_mw")
    ]
    if flow_cols:
        net_flow = df[flow_cols].sum(axis=1)
        df["interchange_delta_mw"] = net_flow.diff(1).abs()
    else:
        df["interchange_delta_mw"] = np.nan

    # 4. % renovables no gestionables sobre demanda
    renewables_cols = [c for c in df.columns if c in ("wind_mw", "solar_pv_mw", "solar_thermal_mw")]
    if renewables_cols and "demand_real_mw" in df.columns:
        renewables = df[renewables_cols].sum(axis=1)
        df["renewables_pct"] = (
            renewables / df["demand_real_mw"].replace(0, np.nan) * 100
        ).clip(0, 100)
    else:
        df["renewables_pct"] = np.nan

    # 5. Presión compuesta [0-1] — fórmula canónica absoluta (compute_composite_pressure_score)
    # NOTA: se eliminó la normalización min-max rolling 7 días que causaba train/serve skew.
    # Ahora usa los mismos umbrales operacionales absolutos que el runtime y la validación.
    # El dataset DEBE reconstruirse y los modelos DEBEN reentrenarse tras este cambio.
    required_cols = {"balance_mw", "reserve_activated_pct", "interchange_delta_mw", "renewables_pct"}
    if required_cols.issubset(df.columns):
        df["composite_pressure"] = np.vectorize(compute_composite_pressure_score)(
            df["balance_mw"].fillna(0.0),
            df["reserve_activated_pct"].fillna(0.0),
            df["interchange_delta_mw"].fillna(0.0),
            df["renewables_pct"].fillna(0.0),
        )
    else:
        df["composite_pressure"] = np.nan

    return df


# ---------------------------------------------------------------------------
# Feature engineering — Features temporales
# ---------------------------------------------------------------------------

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Añade features temporales con encoding circular para evitar discontinuidades."""
    idx = df.index

    df["hour_sin"] = np.sin(2 * math.pi * idx.hour / 24)
    df["hour_cos"] = np.cos(2 * math.pi * idx.hour / 24)
    df["dow_sin"] = np.sin(2 * math.pi * idx.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * math.pi * idx.dayofweek / 7)
    df["month_sin"] = np.sin(2 * math.pi * idx.month / 12)
    df["month_cos"] = np.cos(2 * math.pi * idx.month / 12)
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)

    # Festivos nacionales de España (requiere holidays>=0.25)
    try:
        import holidays as hols
        es_holidays = hols.Spain(years=range(idx.year.min(), idx.year.max() + 1))
        holiday_dates = pd.DatetimeIndex(list(es_holidays.keys())).tz_localize("UTC")
        df["is_holiday_es"] = idx.normalize().isin(holiday_dates).astype(int)
    except ImportError:
        log.warning("'holidays' no instalado. is_holiday_es = 0. pip install holidays")
        df["is_holiday_es"] = 0

    return df


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def build_dataset(
    esios_dir: Path,
    aemet_dir: Path,
    entsoe_dir: Path,
    out_dir: Path,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Path:
    esios = load_esios(esios_dir)
    aemet = load_aemet(aemet_dir)
    entsoe = load_entsoe(entsoe_dir)

    available = [df for df in (esios, aemet, entsoe) if not df.empty]
    if not available:
        raise RuntimeError(
            "No hay datos disponibles. Ejecutar primero download_esios.py, "
            "download_aemet.py y download_entsoe.py."
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

    log.info("Calculando proxies de estabilidad…")
    dataset = compute_stability_proxies(dataset)

    log.info("Añadiendo features temporales…")
    dataset = add_temporal_features(dataset)

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construye dataset de entrenamiento")
    parser.add_argument("--esios", default="data/raw/esios")
    parser.add_argument("--aemet", default="data/raw/aemet")
    parser.add_argument("--entsoe", default="data/raw/entsoe")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--start", default=None, help="Filtro inicio (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="Filtro fin (YYYY-MM-DD)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_dataset(
        esios_dir=Path(args.esios),
        aemet_dir=Path(args.aemet),
        entsoe_dir=Path(args.entsoe),
        out_dir=Path(args.out),
        start=args.start,
        end=args.end,
    )
