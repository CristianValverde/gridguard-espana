"""
ml/data/download_esios.py — Descarga histórico ESIOS (Red Eléctrica de España).

Fuente: https://api.esios.ree.es/
Cobertura: 2014-01-01 hasta hoy (granularidad horaria y 10 minutos).
Credencial: variable de entorno ESIOS_TOKEN (nunca hardcodeada).

Uso:
    python -m ml.data.download_esios --start 2014-01-01 --end 2024-12-31 --out data/raw/esios
    python -m ml.data.download_esios --start 2024-01-01                    # hasta hoy

Salidas (parquet, particionadas por año):
    data/raw/esios/demand/demand_{year}.parquet
    data/raw/esios/generation/generation_{year}.parquet
    data/raw/esios/interchange/interchange_{year}.parquet
    data/raw/esios/reserve/reserve_{year}.parquet
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import date, datetime, timedelta
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
log = logging.getLogger("download_esios")

BASE_URL = "https://api.esios.ree.es"

# Indicadores ESIOS por categoría.
# IDs verificados contra la API en producción (junio 2026).
INDICATORS: dict[str, dict[str, int]] = {
    "demand": {
        "demand_real_mw": 1293,            # Demanda real del sistema (10 min)
    },
    "generation": {
        "nuclear_mw": 1153,                # Generación medida Nuclear
        "hydro_mw": 10035,                 # Generación medida Hidráulica (total)
        "wind_mw": 10037,                  # Generación medida Eólica (total)
        "solar_thermal_mw": 1162,          # Generación medida Solar térmica
        "solar_pv_mw": 1161,               # Generación medida Solar fotovoltaica
        "gas_cc_mw": 1156,                 # Generación medida Ciclo combinado
        "coal_mw": 10036,                  # Generación medida Carbón
        "cogeneration_mw": 10039,          # Generación medida Cogeneración
        "pumping_mw": 1172,                # Generación medida Consumo bombeo (negativo)
    },
    "interchange": {
        "es_fr_mw": 10045,                 # Saldo Francia (positivo = importación a ES)
        "es_pt_mw": 10044,                 # Saldo Portugal
        "es_ma_mw": 10046,                 # Saldo Marruecos
    },
    "reserve": {
        "secondary_reserve_up_mwh": 680,   # Energía activada regulación secundaria subir
        "secondary_reserve_dn_mwh": 681,   # Energía activada regulación secundaria bajar
    },
}

TIME_TRUNC = {
    "demand": "ten_minutes",
    "generation": "hour",
    "interchange": "hour",
    "reserve": "hour",
}


def _get_token() -> str:
    token = os.environ.get("ESIOS_TOKEN")
    if not token:
        raise EnvironmentError(
            "ESIOS_TOKEN no definida. Solicitar en consultasios@ree.es"
        )
    return token


def _headers(token: str) -> dict:
    return {
        "Accept": "application/json; application/vnd.esios-api-v2+json",
        "Content-Type": "application/json",
        "Authorization": f'Token token="{token}"',
        "x-api-key": token,
    }


def _fetch_indicator(
    indicator_id: int,
    start: date,
    end: date,
    time_trunc: str,
    token: str,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Descarga un indicador ESIOS con retry exponencial.

    Returns:
        DataFrame con columnas ['timestamp', 'value'] indexado por timestamp UTC.
    """
    url = f"{BASE_URL}/indicators/{indicator_id}"
    params = {
        "start_date": f"{start}T00:00:00Z",
        "end_date": f"{end}T23:59:59Z",
        "time_trunc": time_trunc,
    }
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(
                url, headers=_headers(token), params=params, timeout=60
            )
            resp.raise_for_status()
            data = resp.json()
            values = data["indicator"]["values"]
            if not values:
                log.warning("Indicador %d: respuesta vacía para %s → %s", indicator_id, start, end)
                return pd.DataFrame(columns=["timestamp", "value"])
            df = pd.DataFrame(values)
            df["timestamp"] = pd.to_datetime(df["datetime"], utc=True)
            # Sumar valores de todos los geo_ids para cada timestamp
            # (múltiples unidades: reactores nucleares, centrales hidráulicas…)
            df = (
                df.groupby("timestamp")["value"]
                .sum()
                .reset_index()
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
            return df
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 429:
                wait = 2 ** attempt * 5
                log.warning("Rate limit en indicador %d. Esperando %ds…", indicator_id, wait)
                time.sleep(wait)
            elif attempt == max_retries:
                log.error("Error HTTP en indicador %d tras %d intentos: %s", indicator_id, max_retries, exc)
                raise
            else:
                time.sleep(2 ** attempt)
        except requests.RequestException as exc:
            if attempt == max_retries:
                log.error("Error de red en indicador %d: %s", indicator_id, exc)
                raise
            time.sleep(2 ** attempt)
    return pd.DataFrame(columns=["timestamp", "value"])


def _date_chunks(start: date, end: date, chunk_days: int) -> list[tuple[date, date]]:
    """Divide un rango de fechas en segmentos de chunk_days días."""
    chunks = []
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)
        chunks.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)
    return chunks


def download_category(
    category: str,
    start: date,
    end: date,
    out_dir: Path,
    token: str,
) -> Path:
    """Descarga todos los indicadores de una categoría y los guarda en parquet.

    Para datos 10-minuto (demanda) usa chunks mensuales para evitar timeouts.
    """
    indicators = INDICATORS[category]
    time_trunc = TIME_TRUNC[category]

    # Chunk mensual para todos: evita 504 en indicadores con muchos geo_ids (hidro, eólica…)
    chunk_days = 28

    series_parts: dict[str, list[pd.Series]] = {col: [] for col in indicators}

    for chunk_start, chunk_end in _date_chunks(start, end, chunk_days):
        for col_name, ind_id in indicators.items():
            log.info(
                "  Descargando %s (id=%d) [%s → %s]",
                col_name, ind_id, chunk_start, chunk_end,
            )
            df = _fetch_indicator(ind_id, chunk_start, chunk_end, time_trunc, token)
            if not df.empty:
                series_parts[col_name].append(
                    df.set_index("timestamp")["value"].rename(col_name)
                )
            time.sleep(0.5)

    frames = {}
    for col_name, parts in series_parts.items():
        if parts:
            frames[col_name] = pd.concat(parts).sort_index()

    if not frames:
        log.warning("Categoría %s: sin datos para %s → %s", category, start, end)
        return out_dir

    merged = pd.concat(frames.values(), axis=1)
    merged.index = pd.DatetimeIndex(merged.index, name="timestamp")

    out_dir.mkdir(parents=True, exist_ok=True)
    year = start.year
    out_path = out_dir / f"{category}_{year}.parquet"
    merged.to_parquet(out_path, engine="pyarrow", compression="snappy")
    log.info("  Guardado: %s (%d filas)", out_path, len(merged))
    return out_path


def run(
    start: date,
    end: date,
    out_dir: Path,
    categories: Optional[list[str]] = None,
) -> None:
    token = _get_token()
    cats = categories or list(INDICATORS.keys())

    # Iterar por año para archivos manejables
    current_year = start.year
    end_year = end.year

    while current_year <= end_year:
        year_start = max(start, date(current_year, 1, 1))
        year_end = min(end, date(current_year, 12, 31))
        log.info("=== Año %d [%s → %s] ===", current_year, year_start, year_end)

        for cat in cats:
            log.info("  Categoría: %s", cat)
            download_category(
                cat, year_start, year_end, out_dir / cat, token
            )
        current_year += 1

    log.info("Descarga ESIOS completada. Archivos en: %s", out_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Descarga histórico ESIOS")
    parser.add_argument(
        "--start", default="2014-01-01",
        help="Fecha inicio (YYYY-MM-DD). Default: 2014-01-01"
    )
    parser.add_argument(
        "--end", default=str(date.today()),
        help="Fecha fin (YYYY-MM-DD). Default: hoy"
    )
    parser.add_argument(
        "--out", default="data/raw/esios",
        help="Directorio de salida"
    )
    parser.add_argument(
        "--categories", nargs="*",
        choices=list(INDICATORS.keys()),
        help="Categorías a descargar. Default: todas"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
        out_dir=Path(args.out),
        categories=args.categories,
    )
