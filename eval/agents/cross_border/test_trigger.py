"""Trigger offline del CrossBorderAgent: las tools que puede disparar son
exactamente las diseñadas, y sus servers conectados las sirven."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from guardrails.hooks import _ALLOWED_TOOLS

pytestmark = pytest.mark.agent_cross_border

_EXPECTED_TOOLS = {
    "get_cross_border_flows", "get_capacity_margin",
    "get_eu_generation", "get_mitigation_procedure",
    "get_eu_market_prices",
}
_EXPECTED_SERVERS = {"GRID_DATA_TOOLSET", "KB_TOOLSET"}


def test_whitelist_is_exactly_expected():
    assert _ALLOWED_TOOLS["cross_border_agent"] == _EXPECTED_TOOLS


def test_connected_servers_serve_the_whitelist():
    # Task 2: los toolsets son singletons compartidos importados de
    # agents/mcp_toolsets.py (ya no hay cadenas mcp_servers.* en el agente).
    src = (Path(__file__).resolve().parents[3] / "agents" / "cross_border.py").read_text(encoding="utf-8")
    for server in _EXPECTED_SERVERS:
        assert server in src, f"server no conectado: {server}"


# ---------------------------------------------------------------------------
# Task 11: trigger + trayectoria live (integration)
#
# NOTA: el brief asumia `CrossBorderAgent.agent` (patron AgentTool), pero
# agents/cross_border.py exporta `cross_border_agent`, una instancia Agent
# directa (no un AgentTool) -- verificado leyendo el modulo fuente. Se usa
# `cross_border_agent` tal cual, sin `.agent`.
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestCrossBorderLiveTriggers:
    _PROMPT_EMERGENCY = (
        "[EVAL] mode='emergency'. Evalua si Espana puede importar potencia de "
        "emergencia ahora mismo. Snapshot: "
        + json.dumps({"demand_mw": 30200, "blackout_probability": 0.72,
                      "capacity_margin_pct": 3})
    )

    _PROMPT_ASSESSMENT = (
        "[EVAL] mode='assessment'. Evalua proactivamente la capacidad de "
        "importacion disponible sin urgencia. Snapshot: "
        + json.dumps({"demand_mw": 30200, "blackout_probability": 0.15,
                      "capacity_margin_pct": 35})
    )

    @pytest.mark.flaky(reruns=1, reruns_delay=5)
    def test_low_margin_scenario_triggers_mitigation(self):
        from agents.cross_border import cross_border_agent
        from eval.agents._harness import run_agent_live
        result = run_agent_live(cross_border_agent, self._PROMPT_EMERGENCY)
        tools = [step["tool"] for step in result["trajectory"]]
        assert "get_cross_border_flows" in tools, "trigger de flujos ausente"
        assert "get_capacity_margin" in tools, "trigger de margen de capacidad ausente"
        assert "get_mitigation_procedure" in tools, "trigger de procedimiento de mitigacion ausente"

    @pytest.mark.flaky(reruns=1, reruns_delay=5)
    def test_ample_margin_scenario_skips_mitigation(self):
        from agents.cross_border import cross_border_agent
        from eval.agents._harness import run_agent_live
        result = run_agent_live(cross_border_agent, self._PROMPT_ASSESSMENT)
        tools = [step["tool"] for step in result["trajectory"]]
        assert "get_cross_border_flows" in tools, "trigger de flujos ausente"
        assert "get_mitigation_procedure" not in tools, (
            "no deberia buscar procedimiento de mitigacion con margen holgado"
        )
