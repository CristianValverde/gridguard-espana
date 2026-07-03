"""Trigger offline del ReportAgent: las tools que puede disparar son
exactamente las diseñadas, y sus servers conectados las sirven."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from guardrails.hooks import _ALLOWED_TOOLS

pytestmark = pytest.mark.agent_report

_EXPECTED_TOOLS = {
    "query_incidents", "get_risk_history", "get_blackout_probability",
    "find_similar_incident", "write_report",
    "get_markov_trajectory", "get_state_transitions",
}
_EXPECTED_SERVERS = {
    "MEMORY_TOOLSET", "ML_TOOLSET",
    "KB_TOOLSET",
}


def test_whitelist_is_exactly_expected():
    assert _ALLOWED_TOOLS["report_agent"] == _EXPECTED_TOOLS


def test_connected_servers_serve_the_whitelist():
    # Task 2: los toolsets son singletons compartidos importados de
    # agents/mcp_toolsets.py (ya no hay cadenas mcp_servers.* en el agente).
    src = (Path(__file__).resolve().parents[3] / "agents" / "report_agent.py").read_text(encoding="utf-8")
    for server in _EXPECTED_SERVERS:
        assert server in src, f"server no conectado: {server}"


# ---------------------------------------------------------------------------
# Task 11: trigger + trayectoria live (integration)
#
# NOTA: el brief asumia `ReportAgent.agent` (patron AgentTool), pero
# agents/report_agent.py exporta `report_agent`, una instancia Agent directa
# (no un AgentTool) -- verificado leyendo el modulo fuente. Se usa
# `report_agent` tal cual, sin `.agent`.
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestReportAgentLiveTriggers:
    _PROMPT = (
        "[EVAL] Genera un informe de incidente para este estado de red critico. "
        "report_type='emergency'. Snapshot: "
        + json.dumps({"demand_mw": 30200, "renewables_pct": 74.0,
                      "blackout_probability": 0.72, "risk_score": 85})
    )

    def test_consults_history_before_writing(self):
        from agents.report_agent import report_agent
        from eval.agents._harness import run_agent_live
        result = run_agent_live(report_agent, self._PROMPT)
        tools = [step["tool"] for step in result["trajectory"]]
        assert "find_similar_incident" in tools, "trigger RAG ausente"
        assert "get_markov_trajectory" in tools, "trigger Markov ausente"
        if "write_report" in tools:
            assert tools.index("find_similar_incident") < tools.index("write_report"), (
                "trayectoria invalida: write_report antes de consultar historicos"
            )
