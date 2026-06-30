"""
ml/data/download_aemet.py — Descarga histórico AEMET OpenData.

Fuente: https://opendata.aemet.es/opendata/api/
Cobertura: ~2000 hasta hoy (observaciones horarias de estaciones sinópticas).
Credencial: variable de entorno AEMET_API_KEY (nunca hardcodeada).

Estrategia de descarga:
  AEMET no permite rangos largos en una sola llamada. Se descarga mes a mes.
  El endpoint devuelve primero una URL de datos que se consulta en segundo paso.

Estaciones seleccionadas (representativas de las zonas de generación renovable):
  - 3195 · Madrid (Retiro)         — demanda/temperatura referencia
  - 0076 · Bilbao (Sondika)        — zona eólica norte
  - 5514 · Sevilla (Tablada)       — zona solar sur
  - 9898 · Zaragoza (Aeropuerto)   — zona eólica interior
  - 8025 · Valencia                — zona solar levante
  - 1387 · Santiago (Labacolla)    — zona eólica Galicia
  - 7031 · Murcia                  — zona solar sureste

Uso:
    python -m ml.data.download_aemet --start 2000-01-01 --end 2024-12-31 --out data/raw/aemet
    python -m ml.data.download_aemet --start 2020-01-01  # hasta hoy, estaciones default

Salidas:
    data/raw/aemet/observations_{year}.parquet
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import date
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("download_aemet")

BASE_URL = "https://opendata.aemet.es/opendata/api"

DEFAULT_STATIONS = [
    "3195",  # Madrid Retiro
    "1082",  # Bilbao/Santander norte
    "5783",  # Sevilla San Pablo
    "9898",  # Zaragoza Aeropuerto
    "8416Y", # Valencia Aeropuerto
    "1387",  # Santiago de Compostela
    "7031",  # Murcia
]

# Variables de interés para features de generación renovable
# Formato de la API /valores/climatologicos/diarios (datos diarios, no horarios)
KEEP_VARS = [
    "indicativo",  # código estación (AEMET usa 'indicativo', no 'idema')
    "fecha",       # fecha (YYYY-MM-DD)
    "tmed",        # temperatura media (°C)
    "tmax",        # temperatura máxima (°C)
    "tmin",        # temperatura mínima (°C)
    "velmedia",    # velocidad media viento (m/s)
    "racha",       # racha máxima viento (m/s)
    "dir",         # dirección viento dominante (°/10)
    "prec",        # precipitación total (mm)
    "presMax",     # presión máxima (hPa)
    "presMin",     # presión mínima (hPa)
    "hrMedia",     # humedad relativa media (%)
    "hrMax",       # humedad relativa máxima (%)
    "hrMin",       # humedad relativa mínima (%)
]


def _get_api_key() -> str:
    key = os.environ.get("AEMET_API_KEY")
    if not key:
        raise EnvironmentError(
            "AEMET_API_KEY no definida. Obtener en opendata.aemet.es"
        )
    return key


def _fetch_json(url: str, api_key: str, max_retries: int = 3) -> dict | list | None:
    """GET genérico con retry y apikey."""
    params = {"api_key": api_key}
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 429:
                wait = 2 ** attempt * 10
                log.warning("Rate limit AEMET. Esperando %ds…", wait)
                time.sleep(wait)
            elif attempt == max_retries:
                log.error("Error HTTP en %s: %s", url, exc)
                return None
            else:
                time.sleep(2 ** attempt)
        except requests.RequestException as exc:
            if attempt == max_retries:
                log.error("Error de red en %s: %s", url, exc)
                return None
            time.sleep(2 ** attempt)
    return None


def _fetch_observations_month(
    station_id: str,
    year: int,
    month: int,
    api_key: str,
) -> pd.DataFrame:
    """Descarga datos climatológicos diarios de una estación para un mes completo.

    Usa el endpoint /valores/climatologicos/diarios que da resúmenes diarios
    (no horarios) con temperatura, viento, precipitación y presión.

    AEMET usa un patrón de dos pasos:
    1. Llamada al endpoint → devuelve {'datos': url_datos, 'estado': 200}
    2. GET url_datos → JSON con los registros diarios
    """
    from calendar import monthrange
    last_day = monthrange(year, month)[1]
    date_ini = f"{year}-{month:02d}-01T00:00:00UTC"
    date_end = f"{year}-{month:02d}-{last_day:02d}T23:59:59UTC"

    # Endpoint correcto para histórico: /valores/climatologicos/diarios
    # Nota: fechaini/fechafin van ANTES de estacion en esta ruta
    endpoint = (
        f"{BASE_URL}/valores/climatologicos/diarios/datos/"
        f"fechaini/{date_ini}/fechafin/{date_end}/estacion/{station_id}"
    )

    meta = _fetch_json(endpoint, api_key)
    if not meta or meta.get("estado") != 200:
        log.debug("Sin datos para estación %s en %d-%02d", station_id, year, month)
        return pd.DataFrame()

    datos_url = meta.get("datos")
    if not datos_url:
        return pd.DataFrame()

    # Segundo paso: obtener los datos reales
    try:
        resp = requests.get(datos_url, timeout=120)
        resp.raise_for_status()
        obs = resp.json()
    except Exception as exc:
        log.warning("Error al obtener datos de %s: %s", datos_url, exc)
        return pd.DataFrame()

    if not obs:
        return pd.DataFrame()

    df = pd.DataFrame(obs)

    # Seleccionar solo columnas disponibles de las que nos interesan
    available = [c for c in KEEP_VARS if c in df.columns]
    df = df[available].copy()

    # Convertir fecha a timestamp UTC (datos diarios: medianoche UTC)
    if "fecha" in df.columns:
        df["timestamp"] = pd.to_datetime(df["fecha"], format="%Y-%m-%d", utc=True)
        df = df.rename(columns={"indicativo": "station_id"})

    # AEMET usa comas decimales en campos numéricos — normalizar a float
    skip_cols = {"station_id", "timestamp", "fecha"}
    for col in df.columns:
        if col not in skip_cols:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .apply(pd.to_numeric, errors="coerce")
            )

    return df


def download_year(
    year: int,
    stations: list[str],
    out_dir: Path,
    api_key: str,
) -> Optional[Path]:
    """Descarga todas las observaciones de un año para las estaciones dadas."""
    all_frames: list[pd.DataFrame] = []

    for station in stations:
        log.info("  Estación %s — año %d", station, year)
        for month in range(1, 13):
            df = _fetch_observations_month(station, year, month, api_key)
            if not df.empty:
                all_frames.append(df)
            time.sleep(0.3)  # cortesía AEMET (50 req/min)

    if not all_frames:
        log.warning("Sin datos AEMET para el año %d", year)
        return None

    combined = pd.concat(all_frames, ignore_index=True)

    sort_col = "timestamp" if "timestamp" in combined.columns else "fecha"
    if sort_col in combined.columns:
        combined = combined.sort_values(sort_col).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"observations_{year}.parquet"
    combined.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
    log.info("  Guardado: %s (%d filas)", out_path, len(combined))
    return out_path


def run(
    start: date,
    end: date,
    out_dir: Path,
    stations: Optional[list[str]] = None,
) -> None:
    api_key = _get_api_key()
    station_list = stations or DEFAULT_STATIONS

    for year in range(start.year, end.year + 1):
        log.info("=== AEMET año %d ===", year)
        download_year(year, station_list, out_dir, api_key)

    log.info("Descarga AEMET completada. Archivos en: %s", out_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Descarga histórico AEMET OpenData")
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default=str(date.today()))
    parser.add_argument("--out", default="data/raw/aemet")
    parser.add_argument(
        "--stations", nargs="*", default=None,
        help="IDs de estaciones AEMET. Default: 7 estaciones representativas"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
        out_dir=Path(args.out),
        stations=args.stations,
    )
