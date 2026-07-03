"""Wiring de MCP servers por agente: qué server conecta cada módulo (fuente: el código)."""
from __future__ import annotations

from pathlib import Path

_AGENTS_DIR = Path(__file__).resolve().parents[2] / "agents"


def _agent_source(module_name: str) -> str:
    return (_AGENTS_DIR / f"{module_name}.py").read_text(encoding="utf-8")


def test_renewables_uses_grid_data_not_ree_grid():
    src = _agent_source("renewables_forecast")
    assert "GRID_DATA_TOOLSET" in src
    assert "mcp_servers.ree_grid_server" not in src


def test_cross_border_uses_grid_data_not_entsoe():
    src = _agent_source("cross_border")
    assert "GRID_DATA_TOOLSET" in src
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
    assert "ML_TOOLSET" in src


def test_markov_server_file_gone():
    servers = Path(__file__).resolve().parents[2] / "mcp_servers"
    assert not (servers / "markov_server.py").exists()


def test_weather_server_has_no_orphan_alerts_tool():
    servers = Path(__file__).resolve().parents[2] / "mcp_servers"
    src = (servers / "weather_server.py").read_text(encoding="utf-8")
    assert "get_aemet_alerts" not in src


def test_modulos_de_servers_solo_en_mcp_toolsets():
    """Ningun agente define McpToolset propio: las cadenas mcp_servers.* viven
    solo en agents/mcp_toolsets.py."""
    for mod in ("orchestrator", "anomaly_detector", "renewables_forecast",
                "regulatory_compliance", "report_agent", "cross_border"):
        src = _agent_source(mod)
        assert "McpToolset(" not in src, mod
    shared_src = _agent_source("mcp_toolsets")
    for module in ("ml_models_server", "memory_server", "knowledge_base_server",
                   "weather_server", "grid_data_server"):
        assert f"mcp_servers.{module}" in shared_src, module
