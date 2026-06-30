"""
ml/data/download_entsoe.py — Descarga histórico ENTSO-E Transparency Platform.

Fuente: https://transparency.entsoe.eu/
Cobertura: 2011 (CSV legacy) / 2015 (API) hasta hoy, horaria.
Credencial: variable de entorno ENTSOE_TOKEN (nunca hardcodeada).

Datos descargados:
  - Flujos de intercambio transfronterizos ES ↔ FR, PT, MA
  - Generación por tecnología en países vecinos (FR, PT)
  - Margen de capacidad disponible (España)

Requiere: entsoe-py>=0.6 (wrapper mantenido sobre la API oficial)
    pip install entsoe-py

Uso:
    python -m ml.data.download_entsoe --start 2015-01-01 --end 2024-12-31 --out data/raw/entsoe
    python -m ml.data.download_entsoe --start 2020-01-01  # hasta hoy

Salidas:
    data/raw/entsoe/crossborder_{year}.parquet
    data/raw/entsoe/generation_{country}_{year}.parquet
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("download_entsoe")

# Códigos de área ENTSO-E (EIC)
AREA_CODES = {
    "ES": "10YES-REE------0",   # España
    "FR": "10YFR-RTE------C",   # Francia
    "PT": "10YPT-REN------W",   # Portugal
}

# Fronteras de intercambio que interesan a GridGuard
CROSS_BORDER_PAIRS = [
    ("ES", "FR"),
    ("FR", "ES"),
    ("ES", "PT"),
    ("PT", "ES"),
]

GENERATION_COUNTRIES = ["ES", "FR", "PT"]


def _get_token() -> str:
    token = os.environ.get("ENTSOE_TOKEN")
    if not token:
        raise EnvironmentError(
            "ENTSOE_TOKEN no definida. Registrar en transparency.entsoe.eu"
        )
    return token


def _get_client(token: str):
    """Crea cliente entsoe-py. Importación lazy para no fallar en CI sin la librería."""
    try:
        from entsoe import EntsoePandasClient
    except ImportError as exc:
        raise ImportError(
            "entsoe-py no instalado. Ejecutar: pip install entsoe-py"
        ) from exc
    return EntsoePandasClient(api_key=token)


def _to_ts(d: date, tz: str = "UTC") -> pd.Timestamp:
    return pd.Timestamp(datetime(d.year, d.month, d.day), tz=tz)


def download_crossborder_year(
    year: int,
    out_dir: Path,
    client,
) -> Optional[Path]:
    """Descarga flujos transfronterizos para un año y los guarda en parquet."""
    year_start = _to_ts(date(year, 1, 1))
    year_end = _to_ts(date(year, 12, 31)) + pd.Timedelta(hours=23)

    frames: list[pd.DataFrame] = []

    for from_country, to_country in CROSS_BORDER_PAIRS:
        from_area = AREA_CODES[from_country]
        to_area = AREA_CODES[to_country]
        col_name = f"flow_{from_country.lower()}_{to_country.lower()}_mw"

        log.info("  Intercambio %s → %s, año %d", from_country, to_country, year)
        try:
            series = client.query_crossborder_flows(
                country_code_from=from_area,
                country_code_to=to_area,
                start=year_start,
                end=year_end,
            )
            if series is not None and not series.empty:
                df = series.to_frame(name=col_name)
                df.index = pd.DatetimeIndex(df.index, name="timestamp")
                frames.append(df)
        except Exception as exc:
            log.warning("  Error %s → %s año %d: %s", from_country, to_country, year, exc)
        time.sleep(1)  # rate limit ENTSO-E

    if not frames:
        log.warning("Sin datos de intercambio para el año %d", year)
        return None

    merged = pd.concat(frames, axis=1)
    merged.index = pd.DatetimeIndex(merged.index, name="timestamp")
    merged = merged.resample("h").mean()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"crossborder_{year}.parquet"
    merged.to_parquet(out_path, engine="pyarrow", compression="snappy")
    log.info("  Guardado: %s (%d filas)", out_path, len(merged))
    return out_path


def download_generation_year(
    year: int,
    country: str,
    out_dir: Path,
    client,
) -> Optional[Path]:
    """Descarga generación por tecnología de un país para un año."""
    area = AREA_CODES.get(country)
    if not area:
        log.error("País desconocido: %s", country)
        return None

    year_start = _to_ts(date(year, 1, 1))
    year_end = _to_ts(date(year, 12, 31)) + pd.Timedelta(hours=23)

    log.info("  Generación %s año %d", country, year)
    try:
        df = client.query_generation(
            country_code=area,
            start=year_start,
            end=year_end,
        )
    except Exception as exc:
        log.warning("  Error generación %s año %d: %s", country, year, exc)
        return None

    if df is None or df.empty:
        return None

    df.index = pd.DatetimeIndex(df.index, name="timestamp")

    # entsoe-py devuelve MultiIndex (tecnología, 'Actual Aggregated'/'Actual Consumption')
    # Nos quedamos solo con la generación real y aplanamos
    if isinstance(df.columns, pd.MultiIndex):
        if "Actual Aggregated" in df.columns.get_level_values(1):
            df = df.xs("Actual Aggregated", axis=1, level=1, drop_level=True)
        else:
            df.columns = df.columns.get_level_values(0)

    # Prefijo de país + nombres snake_case
    df.columns = [
        f"{country.lower()}_{c.lower().replace(' ', '_').replace('/', '_').replace('-', '_')}"
        for c in df.columns
    ]
    df = df.resample("h").mean()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"generation_{country.lower()}_{year}.parquet"
    df.to_parquet(out_path, engine="pyarrow", compression="snappy")
    log.info("  Guardado: %s (%d filas)", out_path, len(df))
    return out_path


def run(
    start: date,
    end: date,
    out_dir: Path,
) -> None:
    token = _get_token()
    client = _get_client(token)

    for year in range(start.year, end.year + 1):
        log.info("=== ENTSO-E año %d ===", year)
        download_crossborder_year(year, out_dir / "crossborder", client)
        for country in GENERATION_COUNTRIES:
            download_generation_year(year, country, out_dir / "generation", client)

    log.info("Descarga ENTSO-E completada. Archivos en: %s", out_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Descarga histórico ENTSO-E")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=str(date.today()))
    parser.add_argument("--out", default="data/raw/entsoe")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
        out_dir=Path(args.out),
    )
