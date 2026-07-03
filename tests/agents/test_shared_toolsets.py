"""Toolsets MCP singleton compartidos: 5 instancias, una por server.

Spec: docs/superpowers/specs/2026-07-03-mcp-shared-transport-design.md
"""
from __future__ import annotations

from google.adk.tools.mcp_tool.mcp_toolset import McpToolset

EXPECTED = {
    "ML_TOOLSET": "mcp_servers.ml_models_server",
    "MEMORY_TOOLSET": "mcp_servers.memory_server",
    "KB_TOOLSET": "mcp_servers.knowledge_base_server",
    "WEATHER_TOOLSET": "mcp_servers.weather_server",
    "GRID_DATA_TOOLSET": "mcp_servers.grid_data_server",
}


def test_expone_cinco_toolsets_mcp():
    from agents import mcp_toolsets

    for name in EXPECTED:
        assert isinstance(getattr(mcp_toolsets, name), McpToolset), name


def test_cada_toolset_apunta_a_su_server_con_timeout_60():
    from agents import mcp_toolsets

    for name, module in EXPECTED.items():
        toolset = getattr(mcp_toolsets, name)
        params = toolset._connection_params
        assert params.timeout == 60.0, name
        assert params.server_params.command == "python", name
        assert params.server_params.args == ["-m", module], name


def test_las_cinco_instancias_son_distintas():
    from agents import mcp_toolsets

    ids = {id(getattr(mcp_toolsets, name)) for name in EXPECTED}
    assert len(ids) == 5
