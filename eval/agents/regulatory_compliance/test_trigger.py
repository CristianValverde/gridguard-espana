"""Trigger offline del RegulatoryComplianceSkill: las tools que puede disparar son
exactamente las diseñadas, y sus servers conectados las sirven."""
from __future__ import annotations

from pathlib import Path

import pytest

from guardrails.hooks import _ALLOWED_TOOLS

pytestmark = pytest.mark.agent_regulatory

_EXPECTED_TOOLS = {
    "search_regulations", "get_grid_state",
}
_EXPECTED_SERVERS = {"mcp_servers.knowledge_base_server", "mcp_servers.ml_models_server"}


def test_whitelist_is_exactly_expected():
    assert _ALLOWED_TOOLS["regulatory_compliance_skill"] == _EXPECTED_TOOLS


def test_connected_servers_serve_the_whitelist():
    src = (Path(__file__).resolve().parents[3] / "agents" / "regulatory_compliance.py").read_text(encoding="utf-8")
    for server in _EXPECTED_SERVERS:
        assert server in src, f"server no conectado: {server}"
