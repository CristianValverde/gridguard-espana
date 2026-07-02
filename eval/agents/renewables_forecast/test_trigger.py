"""Trigger offline del RenewablesForecastSkill: las tools que puede disparar son
exactamente las diseñadas, y sus servers conectados las sirven."""
from __future__ import annotations

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
    "mcp_servers.weather_server", "mcp_servers.grid_data_server",
    "mcp_servers.ml_models_server", "mcp_servers.memory_server",
}


def test_whitelist_is_exactly_expected():
    assert _ALLOWED_TOOLS["renewables_forecast_skill"] == _EXPECTED_TOOLS


def test_connected_servers_serve_the_whitelist():
    src = (Path(__file__).resolve().parents[3] / "agents" / "renewables_forecast.py").read_text(encoding="utf-8")
    for server in _EXPECTED_SERVERS:
        assert server in src, f"server no conectado: {server}"
