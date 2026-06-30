"""
mcp_servers/ree_grid_server.py — MCP: ree-grid

Wrapper autenticado sobre ESIOS API con caché en memoria (TTL 9 min).
Credencial: ESIOS_TOKEN (variable de entorno, nunca hardcodeada).

Herramientas expuestas:
  get_realtime_demand(region)          -> DemandSnapshot (JSON)
  get_generation_mix(start, end)       -> GenerationMix (JSON)
  get_interchange_flows(border)        -> InterchangeFlow (JSON)
  get_programmed_generation()          -> ProgrammedGeneration (JSON)
  get_reserve_activation()             -> ReserveStatus (JSON)

Iniciar servidor:
    python -m mcp_servers.ree_grid_server
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")
from datetime import datetime, timezone
from typing import Any

import requests
from mcp.server.fastmcp import FastMCP

from schemas.models import (
    DemandSnapshot, GenerationMix, InterchangeFlow,
    ProgrammedGeneration, ReserveStatus,
)

mcp = FastMCP("gridguard-ree-grid")

BASE_URL = "https://api.esios.ree.es"
CACHE_TTL = 540  # 9 minutos en segundos

# {cache_key: (data, expiry_unix_ts)}
_cache: dict[str, tuple[Any, float]] = {}

# Indicadores ESIOS por herramienta (mismos IDs que download_esios.py)
_INDICATORS = {
    "demand_real": 1293,
    "demand_programmed": 10034,
    "nuclear": 460, "hydro": 1163, "wind": 551,
    "solar_thermal": 1160, "solar_pv": 1161,
    "gas_cc": 545, "coal": 542, "cogeneration": 544,
    "interchange_es_fr": 10209, "interchange_es_pt": 10210, "interchange_es_ma": 10211,
    "reserve_secondary": 785, "reserve_tertiary_up": 786, "reserve_tertiary_dn": 787,
}

_BORDER_MAP = {
    "ES-FR": "interchange_es_fr",
    "ES-PT": "interchange_es_pt",
    "ES-MA": "interchange_es_ma",
}


def _token() -> str:
    t = os.environ.get("ESIOS_TOKEN")
    if not t:
        raise EnvironmentError("ESIOS_TOKEN no definida")
    return t


def _headers() -> dict:
    return {
        "Accept": "application/json; application/vnd.esios-api-v2+json",
        "Authorization": f'Token token="{_token()}"',
    }


def _fetch_indicator(indicator_id: int, time_trunc: str = "hour") -> list[dict]:
    """Llama a ESIOS para la última hora disponible."""
    now = datetime.now(timezone.utc)
    start = now.strftime("%Y-%m-%dT%H:00:00Z")
    end = now.strftime("%Y-%m-%dT%H:59:59Z")
    url = f"{BASE_URL}/indicators/{indicator_id}"
    resp = requests.get(
        url,
        headers=_headers(),
        params={"start_date": start, "end_date": end, "time_trunc": time_trunc},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["indicator"]["values"]


def _latest_value(values: list[dict]) -> float:
    """Devuelve el valor más reciente de una lista de valores ESIOS."""
    if not values:
        return 0.0
    return float(sorted(values, key=lambda x: x["datetime"])[-1].get("value", 0) or 0)


def _cached(key: str, build_fn):
    """Envuelve build_fn con caché TTL."""
    now = time.monotonic()
    if key in _cache and _cache[key][1] > now:
        return _cache[key][0]
    result = build_fn()
    _cache[key] = (result, now + CACHE_TTL)
    return result


# ---------------------------------------------------------------------------
# Herramientas MCP
# ---------------------------------------------------------------------------

@mcp.tool()
def get_realtime_demand(region: str = "peninsular") -> dict:
    """Demanda real del sistema en la hora actual (MW)."""
    def build():
        vals = _fetch_indicator(_INDICATORS["demand_real"], "ten_minutes")
        demand = _latest_value(vals)
        programmed = _latest_value(
            _fetch_indicator(_INDICATORS["demand_programmed"], "hour")
        )
        return DemandSnapshot(
            region=region,
            timestamp=datetime.now(timezone.utc),
            demand_mw=demand,
            programmed_mw=programmed,
        ).model_dump(mode="json")

    return _cached(f"demand_{region}", build)


@mcp.tool()
def get_generation_mix(start: str, end: str) -> dict:
    """Mix de generación por tecnología en un intervalo (MW).

    Args:
        start: ISO 8601 UTC, ej. '2025-04-28T10:00:00Z'
        end:   ISO 8601 UTC, ej. '2025-04-28T11:00:00Z'
    """
    key = f"genmix_{start}_{end}"

    def build():
        tech_map = {
            "nuclear_mw": "nuclear", "hydro_mw": "hydro", "wind_mw": "wind",
            "solar_thermal_mw": "solar_thermal", "solar_pv_mw": "solar_pv",
            "gas_cc_mw": "gas_cc", "coal_mw": "coal", "cogeneration_mw": "cogeneration",
        }
        tech_mw: dict[str, float] = {}
        for col, key_name in tech_map.items():
            ind_id = _INDICATORS[key_name]
            url = f"{BASE_URL}/indicators/{ind_id}"
            resp = requests.get(
                url, headers=_headers(),
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

    return _cached(key, build)


@mcp.tool()
def get_interchange_flows(border: str = "ES-FR") -> dict:
    """Flujo de intercambio en una frontera (MW). Positivo = exportación desde España.

    Args:
        border: 'ES-FR' | 'ES-PT' | 'ES-MA'
    """
    border = border.upper()
    ind_key = _BORDER_MAP.get(border)
    if not ind_key:
        return {"error": f"Frontera desconocida: {border}. Usar ES-FR, ES-PT o ES-MA"}

    def build():
        vals = _fetch_indicator(_INDICATORS[ind_key], "hour")
        flow = _latest_value(vals)
        return InterchangeFlow(
            border=border,
            timestamp=datetime.now(timezone.utc),
            flow_mw=flow,
        ).model_dump(mode="json")

    return _cached(f"interchange_{border}", build)


@mcp.tool()
def get_programmed_generation() -> dict:
    """Generación programada P48 para la hora actual (MW)."""
    def build():
        vals = _fetch_indicator(_INDICATORS["demand_programmed"], "hour")
        scheduled = _latest_value(vals)
        return ProgrammedGeneration(
            timestamp=datetime.now(timezone.utc),
            scheduled_mw=scheduled if scheduled > 0 else 0.0,
            available_units=[],
        ).model_dump(mode="json")

    return _cached("programmed_gen", build)


@mcp.tool()
def get_reserve_activation() -> dict:
    """Estado de las reservas de regulación secundaria y terciaria."""
    def build():
        sec = _latest_value(_fetch_indicator(_INDICATORS["reserve_secondary"], "hour"))
        t_up = _latest_value(_fetch_indicator(_INDICATORS["reserve_tertiary_up"], "hour"))
        t_dn = _latest_value(_fetch_indicator(_INDICATORS["reserve_tertiary_dn"], "hour"))

        max_sec = 3000.0  # MW — capacidad típica de reserva secundaria en España
        return ReserveStatus(
            timestamp=datetime.now(timezone.utc),
            secondary_reserve_mw=abs(sec),
            tertiary_reserve_mw=abs(t_up) + abs(t_dn),
            activated_secondary_pct=min(abs(sec) / max_sec * 100, 100.0),
            activated_tertiary_pct=min((abs(t_up) + abs(t_dn)) / (max_sec * 2) * 100, 100.0),
        ).model_dump(mode="json")

    return _cached("reserve", build)


if __name__ == "__main__":
    mcp.run()
