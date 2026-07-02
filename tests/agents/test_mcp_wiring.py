"""Wiring de MCP servers por agente: qué server conecta cada módulo (fuente: el código)."""
from __future__ import annotations

from pathlib import Path

_AGENTS_DIR = Path(__file__).resolve().parents[2] / "agents"


def _agent_source(module_name: str) -> str:
    return (_AGENTS_DIR / f"{module_name}.py").read_text(encoding="utf-8")


def test_renewables_uses_grid_data_not_ree_grid():
    src = _agent_source("renewables_forecast")
    assert "mcp_servers.grid_data_server" in src
    assert "mcp_servers.ree_grid_server" not in src


def test_cross_border_uses_grid_data_not_entsoe():
    src = _agent_source("cross_border")
    assert "mcp_servers.grid_data_server" in src
    assert "mcp_servers.entsoe_server" not in src


def test_retired_servers_gone():
    servers = Path(__file__).resolve().parents[2] / "mcp_servers"
    assert not (servers / "ree_grid_server.py").exists()
    assert not (servers / "entsoe_server.py").exists()


def test_markov_server_retired_from_all_agents():
    for mod in ("orchestrator", "anomaly_detector", "renewables_forecast",
                "regulatory_compliance", "report_agent"):
        assert "mcp_servers.markov_server" not in _agent_source(mod), mod


def test_regulatory_now_connects_ml_models():
    src = _agent_source("regulatory_compliance")
    assert "mcp_servers.ml_models_server" in src


def test_markov_server_file_gone():
    servers = Path(__file__).resolve().parents[2] / "mcp_servers"
    assert not (servers / "markov_server.py").exists()
