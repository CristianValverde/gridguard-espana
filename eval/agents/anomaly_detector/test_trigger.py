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
    "ML_TOOLSET", "MEMORY_TOOLSET",
    "KB_TOOLSET", "WEATHER_TOOLSET",
}


def test_whitelist_is_exactly_expected():
    assert _ALLOWED_TOOLS["anomaly_detector_skill"] == _EXPECTED_TOOLS


def test_connected_servers_serve_the_whitelist():
    # Task 2: los toolsets son singletons compartidos importados de
    # agents/mcp_toolsets.py (ya no hay cadenas mcp_servers.* en el agente).
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

    @pytest.mark.flaky(reruns=1, reruns_delay=5)
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

    _NORMAL_PROMPT = (
        "[EVAL] Analiza este snapshot de la red electrica espanola y determina "
        "si hay anomalias. Snapshot: "
        + json.dumps({"demand_mw": 28000, "renewables_pct": 35.0,
                      "balance_mw": 50, "reserve_mw": 1500})
    )

    def test_no_regulations_search_on_normal_scenario(self):
        """Con un escenario claramente NORMAL (demanda ~28000 MW, renovables ~35%,
        sin desviaciones), el anomaly_score debe quedar muy por debajo del umbral 0.3
        de la instruction (prompts/anomaly_detector.md, paso 4). Sin implicacion
        regulatoria, el agente no debe consultar la normativa."""
        from agents.anomaly_detector import AnomalyDetectorSkill
        from eval.agents._harness import run_agent_live
        result = run_agent_live(AnomalyDetectorSkill.agent, self._NORMAL_PROMPT)
        tools = [step["tool"] for step in result["trajectory"]]
        assert "search_regulations" not in tools, (
            "en escenario normal, sin anomalia con implicacion regulatoria "
            "(anomaly_score <= 0.3), no debe dispararse search_regulations"
        )
