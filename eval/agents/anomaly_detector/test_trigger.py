"""Trigger offline del AnomalyDetectorSkill: las tools que puede disparar son
exactamente las diseñadas, y sus servers conectados las sirven."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from guardrails.hooks import _ALLOWED_TOOLS

pytestmark = pytest.mark.agent_anomaly

_EXPECTED_TOOLS = {
    "get_anomaly_score", "stability_proxy_calculator",
    "get_baseline", "search_regulations",
    "get_grid_state", "get_wind_forecast", "get_solar_radiation",
}
_EXPECTED_SERVERS = {
    "mcp_servers.ml_models_server", "mcp_servers.memory_server",
    "mcp_servers.knowledge_base_server", "mcp_servers.weather_server",
}


def test_whitelist_is_exactly_expected():
    assert _ALLOWED_TOOLS["anomaly_detector_skill"] == _EXPECTED_TOOLS


def test_connected_servers_serve_the_whitelist():
    src = (Path(__file__).resolve().parents[3] / "agents" / "anomaly_detector.py").read_text(encoding="utf-8")
    for server in _EXPECTED_SERVERS:
        assert server in src, f"server no conectado: {server}"


# ---------------------------------------------------------------------------
# Task 11: trigger + trayectoria live (integration)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestAnomalyDetectorLiveTriggers:
    _PROMPT = (
        "[EVAL] Analiza este snapshot de la red electrica espanola y determina "
        "si hay anomalias. Snapshot: "
        + json.dumps({"demand_mw": 30200, "renewables_pct": 92.0,
                      "balance_mw": -4200, "reserve_mw": 1240})
    )

    def test_consults_ml_and_baseline(self):
        from agents.anomaly_detector import AnomalyDetectorSkill
        from eval.agents._harness import run_agent_live
        result = run_agent_live(AnomalyDetectorSkill.agent, self._PROMPT)
        tools = [step["tool"] for step in result["trajectory"]]
        assert "get_baseline" in tools, "trigger de baseline ausente"
        assert "get_anomaly_score" in tools, "trigger de anomaly score ausente"
        assert tools.index("get_baseline") < tools.index("get_anomaly_score"), (
            "trayectoria invalida: get_anomaly_score antes de get_baseline"
        )
