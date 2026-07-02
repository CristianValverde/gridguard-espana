"""Trigger offline del CrossBorderAgent: las tools que puede disparar son
exactamente las diseñadas, y sus servers conectados las sirven."""
from __future__ import annotations

from pathlib import Path

import pytest

from guardrails.hooks import _ALLOWED_TOOLS

pytestmark = pytest.mark.agent_cross_border

_EXPECTED_TOOLS = {
    "get_cross_border_flows", "get_capacity_margin",
    "get_eu_generation", "get_mitigation_procedure",
    "get_eu_market_prices",
}
_EXPECTED_SERVERS = {"mcp_servers.grid_data_server", "mcp_servers.knowledge_base_server"}


def test_whitelist_is_exactly_expected():
    assert _ALLOWED_TOOLS["cross_border_agent"] == _EXPECTED_TOOLS


def test_connected_servers_serve_the_whitelist():
    src = (Path(__file__).resolve().parents[3] / "agents" / "cross_border.py").read_text(encoding="utf-8")
    for server in _EXPECTED_SERVERS:
        assert server in src, f"server no conectado: {server}"
