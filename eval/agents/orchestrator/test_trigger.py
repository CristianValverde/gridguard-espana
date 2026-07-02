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
    "AnomalyDetectorSkill", "RenewablesForecastSkill",
    "CrossBorderAgent", "ReportAgent",
    "get_grid_state", "store_grid_state", "RegulatoryComplianceSkill",
}
_EXPECTED_SERVERS = {"mcp_servers.ml_models_server", "mcp_servers.memory_server"}


def test_whitelist_is_exactly_expected():
    assert _ALLOWED_TOOLS["gridguard_orchestrator"] == _EXPECTED_TOOLS


def test_connected_servers_serve_the_whitelist():
    src = (Path(__file__).resolve().parents[3] / "agents" / "orchestrator.py").read_text(encoding="utf-8")
    for server in _EXPECTED_SERVERS:
        assert server in src, f"server no conectado: {server}"
