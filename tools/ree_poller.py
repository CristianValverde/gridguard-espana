"""
tools/ree_poller.py — Function Tool: poller de ESIOS API.

Usado directamente por OrchestratorAgent como función Python (sin MCP overhead).
A diferencia del MCP ree-grid, no tiene caché — siempre devuelve datos frescos.
Credencial: ESIOS_TOKEN (variable de entorno).
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone

import requests

log = logging.getLogger(__name__)

BASE_URL = "https://api.esios.ree.es"

_DEMAND_INDICATORS = {
    "demand_real_mw": 1293,
    "demand_programmed_mw": 10034,
}
# IDs verificados contra producción (ml/data/download_esios.py, junio 2026).
# Paridad vigilada por tests/tools/test_ree_poller_ids.py.
_GEN_INDICATORS = {
    "nuclear_mw": 1153, "hydro_mw": 10035, "wind_mw": 10037,
    "solar_thermal_mw": 1162, "solar_pv_mw": 1161,
    "gas_cc_mw": 1156, "coal_mw": 10036, "cogeneration_mw": 10039,
}
_INTERCHANGE_INDICATORS = {
    "es_fr_mw": 10045, "es_pt_mw": 10044, "es_ma_mw": 10046,
}
# sin ID verificado alternativo — mantener
_RESERVE_INDICATORS = {
    "secondary_reserve_mw": 785,
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


def _fetch(indicator_id: int, trunc: str = "hour", retries: int = 3) -> float:
    """Obtiene el último valor disponible de un indicador ESIOS."""
    now = datetime.now(timezone.utc)
    params = {
        "start_date": now.strftime("%Y-%m-%dT%H:00:00Z"),
        "end_date": now.strftime("%Y-%m-%dT%H:59:59Z"),
        "time_trunc": trunc,
    }
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(
                f"{BASE_URL}/indicators/{indicator_id}",
                headers=_headers(), params=params, timeout=20,
            )
            resp.raise_for_status()
            values = resp.json()["indicator"]["values"]
            if not values:
                return 0.0
            return float(sorted(values, key=lambda x: x["datetime"])[-1].get("value") or 0)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 429:
                time.sleep(2 ** attempt * 5)
            elif attempt == retries:
                log.error("Error ESIOS indicador %d: %s", indicator_id, exc)
                return 0.0
            else:
                time.sleep(2 ** attempt)
        except Exception as exc:
            if attempt == retries:
                log.error("Error red indicador %d: %s", indicator_id, exc)
                return 0.0
            time.sleep(2 ** attempt)
    return 0.0


def ree_poller(region: str = "peninsular") -> dict:
    """Consulta ESIOS y devuelve snapshot de la red en tiempo real.

    Usado por OrchestratorAgent para obtener datos frescos en cada ciclo.
    Sin caché — para datos con TTL usar MCP: ree-grid.

    Args:
        region: Zona geográfica ('peninsular', 'canarias', 'baleares')

    Returns:
        dict compatible con GridSnapshot (listo para data_normalizer)
    """
    log.info("Consultando ESIOS API (región=%s)…", region)
    ts = datetime.now(timezone.utc).isoformat()

    generation = {k: _fetch(v) for k, v in _GEN_INDICATORS.items()}
    demand_real = _fetch(_DEMAND_INDICATORS["demand_real_mw"], trunc="ten_minutes")
    demand_prog = _fetch(_DEMAND_INDICATORS["demand_programmed_mw"])
    interchange = {k: _fetch(v) for k, v in _INTERCHANGE_INDICATORS.items()}
    reserve = _fetch(_RESERVE_INDICATORS["secondary_reserve_mw"])

    return {
        "timestamp": ts,
        "region": region,
        "demand_mw": demand_real,
        "generation_by_tech": generation,
        "interchange_flows": interchange,
        "reserve_mw": reserve,
        "programmed_generation_mw": demand_prog,
        "source_timestamp": ts,
    }
