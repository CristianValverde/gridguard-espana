"""
mcp_servers/grid_data_server.py — MCP: grid-data

Datos de red externos en un único server: ESIOS (mix de generación español)
+ ENTSO-E (interconexión europea y precios day-ahead).
Consolida los antiguos ree_grid_server (solo la tool consumida) y entsoe_server
(íntegro). Credenciales: ESIOS_TOKEN y ENTSOE_TOKEN (cada tool usa la suya).

Herramientas expuestas:
  get_generation_mix(market_zone)           -> GenerationMix (JSON)
  get_cross_border_flows(country)           -> list[CrossBorderFlow] (JSON)
  get_eu_generation(country)                -> EUGenerationSnapshot (JSON)
  get_capacity_margin(country)              -> CapacityMargin (JSON)
  get_eu_market_prices(country, start, end) -> list[MarketPrice] (JSON)
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")
from typing import Any

import pandas as pd
import requests
from mcp.server.fastmcp import FastMCP

from schemas.models import (
    CapacityMargin, CrossBorderFlow, EUGenerationSnapshot, GenerationMix, MarketPrice,
)

mcp = FastMCP("gridguard-grid-data")

ESIOS_BASE_URL = "https://api.esios.ree.es"
CACHE_TTL_ESIOS = 540    # 9 min — cadencia ESIOS
CACHE_TTL_ENTSOE = 3600  # 1 h — cadencia ENTSO-E

_cache: dict[str, tuple[Any, float]] = {}


def _cached(key: str, build_fn, ttl: int):
    mono = time.monotonic()
    if key in _cache and _cache[key][1] > mono:
        return _cache[key][0]
    result = build_fn()
    _cache[key] = (result, mono + ttl)
    return result


# ---------------------------------------------------------------------------
# Sección ESIOS — get_generation_mix
# IDs verificados contra producción (ml/data/download_esios.py, junio 2026).
# La paridad la vigila tests/mcp_servers/test_grid_data_server.py.
# ---------------------------------------------------------------------------

_GEN_INDICATORS = {
    "nuclear_mw": 1153, "hydro_mw": 10035, "wind_mw": 10037,
    "solar_thermal_mw": 1162, "solar_pv_mw": 1161,
    "gas_cc_mw": 1156, "coal_mw": 10036, "cogeneration_mw": 10039,
}


def _esios_token() -> str:
    t = os.environ.get("ESIOS_TOKEN")
    if not t:
        raise EnvironmentError("ESIOS_TOKEN no definida")
    return t


def _esios_headers() -> dict:
    # ESIOS exige x-api-key (verificado 2026-07-01 vía api-probe); el esquema
    # legacy `Authorization: Token token="..."` por sí solo devuelve 403.
    token = _esios_token()
    return {
        "Accept": "application/json; application/vnd.esios-api-v2+json",
        "Authorization": f'Token token="{token}"',
        "x-api-key": token,
    }


def _latest_value(values: list[dict]) -> float:
    if not values:
        return 0.0
    return float(sorted(values, key=lambda x: x["datetime"])[-1].get("value", 0) or 0)


@mcp.tool()
def get_generation_mix(start: str, end: str) -> dict:
    """Mix de generación por tecnología en un intervalo (MW).

    Args:
        start: ISO 8601 UTC, ej. '2025-04-28T10:00:00Z'
        end:   ISO 8601 UTC, ej. '2025-04-28T11:00:00Z'
    """
    key = f"genmix_{start}_{end}"

    def build():
        tech_mw: dict[str, float] = {}
        for col, ind_id in _GEN_INDICATORS.items():
            url = f"{ESIOS_BASE_URL}/indicators/{ind_id}"
            resp = requests.get(
                url, headers=_esios_headers(),
                params={"start_date": start, "end_date": end, "time_trunc": "hour"},
                timeout=30,
            )
            if resp.ok:
                vals = resp.json()["indicator"]["values"]
                tech_mw[col] = _latest_value(vals)

        total = sum(tech_mw.values())
        return GenerationMix(
            start=datetime.fromisoformat(start.replace("Z", "+00:00")),
            end=datetime.fromisoformat(end.replace("Z", "+00:00")),
            technology_mw=tech_mw,
            total_mw=total if total > 0 else 1.0,
        ).model_dump(mode="json")

    return _cached(key, build, CACHE_TTL_ESIOS)


# ---------------------------------------------------------------------------
# Sección ENTSO-E — copiada verbatim de entsoe_server.py líneas 37–269
# (AREA_CODES, _entsoe_token, _client, _now_window, get_cross_border_flows,
#  get_eu_generation, get_capacity_margin, _market_prices_impl,
#  _fetch_day_ahead, get_eu_market_prices), con `_cached(key, build)` →
#  `_cached(key, build, CACHE_TTL_ENTSOE)`.
# ---------------------------------------------------------------------------

AREA_CODES = {
    "ES": "10YES-REE------0",
    "FR": "10YFR-RTE------C",
    "PT": "10YPT-REN------W",
}


def _entsoe_token() -> str:
    t = os.environ.get("ENTSOE_TOKEN")
    if not t:
        raise EnvironmentError("ENTSOE_TOKEN no definida")
    return t


def _client():
    try:
        from entsoe import EntsoePandasClient
    except ImportError as exc:
        raise ImportError("entsoe-py no instalado. pip install entsoe-py") from exc
    return EntsoePandasClient(api_key=_entsoe_token())


def _now_window() -> tuple[pd.Timestamp, pd.Timestamp]:
    """Ventana de la última hora en UTC."""
    now = pd.Timestamp.now(tz="UTC").floor("h")
    return now - pd.Timedelta(hours=1), now


# ---------------------------------------------------------------------------
# Herramientas MCP
# ---------------------------------------------------------------------------

@mcp.tool()
def get_cross_border_flows(from_country: str = "ES", to_country: str = "FR") -> dict:
    """Flujo de intercambio transfronterizo en la última hora (MW).

    Args:
        from_country: Código ISO2 del país origen ('ES', 'FR', 'PT')
        to_country:   Código ISO2 del país destino ('ES', 'FR', 'PT')
    """
    from_country = from_country.upper()
    to_country = to_country.upper()

    if from_country not in AREA_CODES:
        return {"error": f"País origen desconocido: {from_country}. Disponibles: {list(AREA_CODES)}"}
    if to_country not in AREA_CODES:
        return {"error": f"País destino desconocido: {to_country}. Disponibles: {list(AREA_CODES)}"}

    key = f"cbf_{from_country}_{to_country}"

    def build():
        try:
            client = _client()
            start, end = _now_window()
            series = client.query_crossborder_flows(
                country_code_from=AREA_CODES[from_country],
                country_code_to=AREA_CODES[to_country],
                start=start, end=end,
            )
            flow_mw = float(series.iloc[-1]) if series is not None and not series.empty else 0.0
            return CrossBorderFlow(
                from_country=from_country,
                to_country=to_country,
                timestamp=datetime.now(timezone.utc),
                flow_mw=flow_mw,
            ).model_dump(mode="json")
        except Exception as exc:
            return {"error": f"Error ENTSO-E cross-border {from_country}→{to_country}: {exc}"}

    return _cached(key, build, CACHE_TTL_ENTSOE)


@mcp.tool()
def get_eu_generation(country: str = "FR") -> dict:
    """Snapshot de generación por tecnología de un país europeo en la última hora.

    Args:
        country: Código ISO2 ('ES', 'FR', 'PT')
    """
    country = country.upper()
    if country not in AREA_CODES:
        return {"error": f"País desconocido: {country}. Disponibles: {list(AREA_CODES)}"}

    def build():
        try:
            client = _client()
            start, end = _now_window()
            df = client.query_generation(
                country_code=AREA_CODES[country],
                start=start, end=end,
            )
            if df is None or df.empty:
                return {"error": f"Sin datos de generación para {country}"}

            # Última fila disponible
            last_row = df.iloc[-1]
            tech_mw = {
                col.lower().replace(" ", "_"): float(v)
                for col, v in last_row.items()
                if pd.notna(v)
            }
            total = sum(tech_mw.values())
            return EUGenerationSnapshot(
                country=country,
                timestamp=datetime.now(timezone.utc),
                technology_mw=tech_mw,
                total_mw=total if total > 0 else 1.0,
            ).model_dump(mode="json")
        except Exception as exc:
            return {"error": f"Error ENTSO-E generation {country}: {exc}"}

    return _cached(f"gen_{country}", build, CACHE_TTL_ENTSOE)


@mcp.tool()
def get_capacity_margin(country: str = "ES") -> dict:
    """Margen de capacidad disponible sobre la demanda pico (%).

    Args:
        country: Código ISO2 ('ES', 'FR', 'PT')
    """
    country = country.upper()
    if country not in AREA_CODES:
        return {"error": f"País desconocido: {country}. Disponibles: {list(AREA_CODES)}"}

    def build():
        try:
            client = _client()
            start, end = _now_window()

            gen_df = client.query_generation(
                country_code=AREA_CODES[country],
                start=start, end=end,
            )
            demand_series = client.query_load(
                country_code=AREA_CODES[country],
                start=start, end=end,
            )

            if gen_df is None or gen_df.empty or demand_series is None or demand_series.empty:
                return {"error": f"Sin datos suficientes para calcular margen en {country}"}

            available_mw = float(gen_df.iloc[-1].sum())
            peak_demand_mw = float(demand_series.iloc[-1])
            margin_pct = (available_mw - peak_demand_mw) / peak_demand_mw * 100 if peak_demand_mw > 0 else 0.0

            return CapacityMargin(
                country=country,
                timestamp=datetime.now(timezone.utc),
                available_capacity_mw=available_mw,
                peak_demand_mw=peak_demand_mw,
                margin_pct=round(margin_pct, 2),
            ).model_dump(mode="json")
        except Exception as exc:
            return {"error": f"Error ENTSO-E capacity margin {country}: {exc}"}

    return _cached(f"margin_{country}", build, CACHE_TTL_ENTSOE)


# ---------------------------------------------------------------------------
# get_eu_market_prices — precios day-ahead A44
#
# get_eu_market_prices: ENTSO-E day-ahead (A44) via entsoe-py EntsoePandasClient.query_day_ahead_prices.
# Confirmado via api-probe 2026-06-30: devuelve pd.Series €/MWh, resolución 15-min (96-97 valores/día),
# índice tz-aware Europe/Paris (FR). CONVERTIR el índice a UTC antes de devolver.
# ---------------------------------------------------------------------------

def _market_prices_impl(countries: list[str], date: str, fetch_fn) -> dict:
    """Lógica pura (sin I/O): precios day-ahead por país para la fecha dada (€/MWh).

    Args:
        countries:  Lista de códigos ISO2. Países no en AREA_CODES se ignoran.
        date:       Fecha en formato YYYY-MM-DD.
        fetch_fn:   Callable(country, date) → pd.Series con índice tz-aware UTC.

    Returns:
        Dict[country → MarketPrice.model_dump(mode='json')].
        Resolución real ENTSO-E: 15-min (96 valores/día), NO horaria.
    """
    out: dict = {}
    for country in countries:
        if country not in AREA_CODES:
            continue
        series = fetch_fn(country, date)
        out[country] = MarketPrice(
            country=country,
            timestamps=list(series.index.to_pydatetime()),
            price_eur_mwh=[float(v) for v in series.values],
        ).model_dump(mode="json")
    return out


def _fetch_day_ahead(country: str, day: str):
    """Transporte real: ENTSO-E A44 via entsoe-py query_day_ahead_prices.

    Probe 2026-06-30: devuelve 96-97 valores/día a resolución 15-min.
    El índice viene en zona local del país (e.g. Europe/Paris para FR);
    se convierte a UTC antes de devolver.
    """
    client = _client()
    start = pd.Timestamp(day, tz="Europe/Madrid")
    end = start + pd.Timedelta(days=1)
    series = client.query_day_ahead_prices(AREA_CODES[country], start=start, end=end)
    # Probe confirmed: index is tz-aware in country local zone → convert to UTC
    series.index = series.index.tz_convert("UTC")
    return series


@mcp.tool()
def get_eu_market_prices(countries: list[str], date: str) -> dict:
    """Precios day-ahead ENTSO-E (€/MWh) para los países indicados en la fecha dada.

    Args:
        countries:  Lista de códigos ISO2 ('ES', 'FR', 'PT').
        date:       Fecha en formato YYYY-MM-DD (ej. '2026-06-30').

    Returns:
        Dict con un MarketPrice por país (timestamps en UTC, price_eur_mwh en €/MWh).
        Resolución: 15-min (96 valores/día), índice UTC.

    Heurística spread: ES-FR > 20 €/MWh → Francia probablemente tiene excedente
    de generación (importación más fácil hacia España).
    """
    return _market_prices_impl(countries, date, fetch_fn=_fetch_day_ahead)


if __name__ == "__main__":
    mcp.run()
