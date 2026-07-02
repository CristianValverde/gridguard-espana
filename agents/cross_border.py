"""
agents/cross_border.py — Sub-agente: CrossBorderAgent.

Tipo: Sub-agent (LLM con loop propio).
Modelo: gemini fast_agent · Temperature 0.2
Invocado por OrchestratorAgent cuando P(blackout) >= 0.30 como medida de mitigación.

Tools disponibles:
  - get_cross_border_flows   (MCP: grid_data_server)
  - get_capacity_margin      (MCP: grid_data_server)
  - get_eu_generation        (MCP: grid_data_server)
  - get_eu_market_prices     (MCP: grid_data_server)
  - get_mitigation_procedure (MCP: knowledge_base_server)

Output: CrossBorderAnalysis — respuesta a "¿podemos importar energía de emergencia/proactivamente?"

Nota: max_turns eliminado en migración a ADK 2.3.0 (el límite de turnos se controla
en el runner, no en LlmAgent). Era max_turns=8 en ADK 1.x.
"""

from __future__ import annotations

from google.adk.agents import Agent
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset, StdioConnectionParams
from google.genai import types as genai_types
from mcp.client.stdio import StdioServerParameters

from config.model_loader import get_model
from agents.instruction_utils import as_static_instruction, load_prompt
from schemas.models import CrossBorderAnalysis

_INSTRUCTION = load_prompt("cross_border")

_grid_data_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="python",
            args=["-m", "mcp_servers.grid_data_server"],
        ),
        timeout=60.0,
    )
)

_kb_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="python",
            args=["-m", "mcp_servers.knowledge_base_server"],
        ),
        timeout=60.0,
    )
)

cross_border_agent = Agent(
    model=get_model("fast_agent"),
    name="cross_border_agent",
    instruction=as_static_instruction(_INSTRUCTION),
    tools=[
        _grid_data_toolset,
        _kb_toolset,
    ],
    output_schema=CrossBorderAnalysis,
    generate_content_config=genai_types.GenerateContentConfig(temperature=0.2),
)
