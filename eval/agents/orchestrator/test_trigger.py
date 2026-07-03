"""Trigger offline del GridGuardOrchestrator: las tools que puede disparar son
exactamente las diseñadas, y sus servers conectados las sirven."""
from __future__ import annotations

from pathlib import Path

import pytest

from guardrails.hooks import _ALLOWED_TOOLS

pytestmark = pytest.mark.agent_orchestrator

_EXPECTED_TOOLS = {
    "ree_poller", "data_normalizer", "stability_proxy_calculator",
    "risk_scorer", "alert_formatter",
    "get_blackout_probability", "store_risk_score",
    "get_serve_features", "get_ensemble_score", "get_blackout_probability_ensemble",
    "anomaly_detector_skill", "renewables_forecast_skill",
    "cross_border_agent", "report_agent",
    "get_grid_state", "store_grid_state", "regulatory_compliance_skill",
}
_EXPECTED_SERVERS = {"ML_TOOLSET", "MEMORY_TOOLSET"}


def test_whitelist_is_exactly_expected():
    assert _ALLOWED_TOOLS["gridguard_orchestrator"] == _EXPECTED_TOOLS


def test_connected_servers_serve_the_whitelist():
    # Task 2: los toolsets son singletons compartidos importados de
    # agents/mcp_toolsets.py (ya no hay cadenas mcp_servers.* en el agente).
    src = (Path(__file__).resolve().parents[3] / "agents" / "orchestrator.py").read_text(encoding="utf-8")
    for server in _EXPECTED_SERVERS:
        assert server in src, f"server no conectado: {server}"


# ---------------------------------------------------------------------------
# Task 11: trigger + trayectoria live (integration) — usa run_scenario
# (no el harness genérico: el orchestrator ya expone su propio entrypoint).
# ---------------------------------------------------------------------------

_SKILL_TOOLS = {"anomaly_detector_skill", "renewables_forecast_skill",
                "regulatory_compliance_skill", "cross_border_agent", "report_agent"}


@pytest.mark.integration
class TestOrchestratorLiveTriggers:
    def test_critical_scenario_triggers_full_escalation(self, blackout_scenario):
        from agents.orchestrator import run_scenario
        result = run_scenario(blackout_scenario)
        tools = {step["tool"] for step in result["trajectory"]}
        missing = _SKILL_TOOLS - tools
        assert not missing, f"Escalación ROJA incompleta; faltan: {missing}"
        assert "alert_formatter" in tools

    def test_normal_scenario_triggers_nothing(self, normal_scenario):
        from agents.orchestrator import run_scenario
        result = run_scenario(normal_scenario)
        tools = {step["tool"] for step in result["trajectory"]}
        fired = _SKILL_TOOLS & tools
        assert not fired, f"Escenario VERDE no debe escalar; disparó: {fired}"
