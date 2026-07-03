"""Trigger offline del RenewablesForecastSkill: las tools que puede disparar son
exactamente las diseñadas, y sus servers conectados las sirven."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from guardrails.hooks import _ALLOWED_TOOLS

pytestmark = pytest.mark.agent_renewables

_EXPECTED_TOOLS = {
    "get_wind_forecast", "get_solar_radiation", "get_generation_mix",
    "get_load_forecast", "get_baseline",
    "get_weather_forecast_24h", "get_grid_state",
}
_EXPECTED_SERVERS = {
    "WEATHER_TOOLSET", "GRID_DATA_TOOLSET",
    "ML_TOOLSET", "MEMORY_TOOLSET",
}


def test_whitelist_is_exactly_expected():
    assert _ALLOWED_TOOLS["renewables_forecast_skill"] == _EXPECTED_TOOLS


def test_connected_servers_serve_the_whitelist():
    # Task 2: los toolsets son singletons compartidos importados de
    # agents/mcp_toolsets.py (ya no hay cadenas mcp_servers.* en el agente).
    src = (Path(__file__).resolve().parents[3] / "agents" / "renewables_forecast.py").read_text(encoding="utf-8")
    for server in _EXPECTED_SERVERS:
        assert server in src, f"server no conectado: {server}"


# ---------------------------------------------------------------------------
# Task 11: trigger + trayectoria live (integration)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestRenewablesForecastLiveTriggers:
    _PROMPT = (
        "[EVAL] Evalua el riesgo operacional de la generacion renovable en las "
        "proximas 24 horas para este estado de red. Snapshot: "
        + json.dumps({"demand_mw": 30200, "wind_mw": 14200, "solar_pv_mw": 13500,
                      "renewables_pct": 92.0})
    )

    def test_consults_weather_and_load_forecast(self):
        from agents.renewables_forecast import RenewablesForecastSkill
        from eval.agents._harness import run_agent_live
        result = run_agent_live(RenewablesForecastSkill.agent, self._PROMPT)
        tools = [step["tool"] for step in result["trajectory"]]
        assert "get_wind_forecast" in tools or "get_solar_radiation" in tools, (
            "trigger meteo (wind/solar) ausente"
        )
        assert "get_load_forecast" in tools, "trigger de load forecast ausente"
