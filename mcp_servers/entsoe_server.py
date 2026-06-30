"""
mcp_servers/entsoe_server.py — MCP: entso-e

Acceso a datos de interconexión europea via ENTSO-E Transparency Platform.
Credencial: ENTSOE_TOKEN (variable de entorno).

Herramientas expuestas:
  get_cross_border_flows(from_country, to_country)  -> CrossBorderFlow (JSON)
  get_eu_generation(country)                         -> EUGenerationSnapshot (JSON)
  get_capacity_margin(country)                       -> CapacityMargin (JSON)

Países soportados: ES, FR, PT (los relevantes para GridGuard)
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
from mcp.server.fastmcp import FastMCP

from schemas.models import CapacityMargin, CrossBorderFlow, EUGenerationSnapshot

mcp = FastMCP("gridguard-entso-e")

CACHE_TTL = 3600  # 1 hora — datos ENTSO-E se actualizan cada hora

_cache: dict[str, tuple[Any, float]] = {}

AREA_CODES = {
    "ES": "10YES-REE------0",
    "FR": "10YFR-RTE------C",
    "PT": "10YPT-REN------W",
}


def _token() -> str:
    t = os.environ.get("ENTSOE_TOKEN")
    if not t:
        raise EnvironmentError("ENTSOE_TOKEN no definida")
    return t


def _client():
    try:
        from entsoe import EntsoePandasClient
    except ImportError as exc:
        raise ImportError("entsoe-py no instalado. pip install entsoe-py") from exc
    return EntsoePandasClient(api_key=_token())


def _now_window() -> tuple[pd.Timestamp, pd.Timestamp]:
    """Ventana de la última hora en UTC."""
    now = pd.Timestamp.now(tz="UTC").floor("h")
    return now - pd.Timedelta(hours=1), now


def _cached(key: str, build_fn):
    mono = time.monotonic()
    if key in _cache and _cache[key][1] > mono:
        return _cache[key][0]
    result = build_fn()
    _cache[key] = (result, mono + CACHE_TTL)
    return result


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

    return _cached(key, build)


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

    return _cached(f"gen_{country}", build)


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

    return _cached(f"margin_{country}", build)


if __name__ == "__main__":
    mcp.run()
