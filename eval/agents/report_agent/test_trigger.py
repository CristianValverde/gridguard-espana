"""Trigger offline del ReportAgent: las tools que puede disparar son
exactamente las diseñadas, y sus servers conectados las sirven."""
from __future__ import annotations

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
    "mcp_servers.memory_server", "mcp_servers.ml_models_server",
    "mcp_servers.knowledge_base_server",
}


def test_whitelist_is_exactly_expected():
    assert _ALLOWED_TOOLS["report_agent"] == _EXPECTED_TOOLS


def test_connected_servers_serve_the_whitelist():
    src = (Path(__file__).resolve().parents[3] / "agents" / "report_agent.py").read_text(encoding="utf-8")
    for server in _EXPECTED_SERVERS:
        assert server in src, f"server no conectado: {server}"
