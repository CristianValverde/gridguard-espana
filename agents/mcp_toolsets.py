"""Toolsets MCP stdio compartidos (un singleton por server).

Todos los agentes importan estas 5 instancias en vez de crear las suyas: el
MCPSessionManager del ADK 2.3.0 cachea la sesion (y su subproceso) por
instancia de McpToolset, asi que compartir instancias reduce los subprocesos
de ~17 (uno por agente x server) a 5 (uno por server) y elimina la tormenta
de imports simultaneos de torch que hacia superar el timeout de 60s en los
evals live.

Spec: docs/superpowers/specs/2026-07-03-mcp-shared-transport-design.md
"""

from __future__ import annotations

from google.adk.tools.mcp_tool.mcp_toolset import McpToolset, StdioConnectionParams
from mcp.client.stdio import StdioServerParameters


def _stdio_toolset(module: str) -> McpToolset:
    return McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command="python",
                args=["-m", module],
            ),
            # 120s: handlers acotados a <=30s, pero los tools sincronos corren
            # inline en el event loop del server (head-of-line blocking) y 60s
            # no absorbia una cola de 2-3 requests (corrida live 2026-07-03).
            timeout=120.0,
        )
    )


ML_TOOLSET = _stdio_toolset("mcp_servers.ml_models_server")
MEMORY_TOOLSET = _stdio_toolset("mcp_servers.memory_server")
KB_TOOLSET = _stdio_toolset("mcp_servers.knowledge_base_server")
WEATHER_TOOLSET = _stdio_toolset("mcp_servers.weather_server")
GRID_DATA_TOOLSET = _stdio_toolset("mcp_servers.grid_data_server")
