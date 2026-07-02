"""
agents/renewables_forecast.py — Skill: RenewablesForecastSkill (AgentTool).

Tipo: AgentTool (síncrono, sin estado entre llamadas).
Modelo: fast_agent · Temperature 0.1
Flujo: Previsiones meteorológicas (6 h y 24 h) + ML demand forecast + Markov → análisis integrado.

Tools disponibles:
  - get_wind_forecast         (MCP: weather)
  - get_solar_radiation       (MCP: weather)
  - get_weather_forecast_24h  (MCP: weather)
  - get_generation_mix        (MCP: ree-grid)
  - get_load_forecast         (MCP: ml-models)
  - get_baseline              (MCP: memory)
  - get_grid_state            (MCP: markov)

Output schema: RenewablesForecast
"""

from __future__ import annotations

from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset, StdioConnectionParams
from google.genai import types as genai_types
from mcp.client.stdio import StdioServerParameters

from config.model_loader import get_model
from agents.instruction_utils import as_static_instruction, load_prompt
from schemas.models import RenewablesForecast

_INSTRUCTION = load_prompt("renewables_forecast")

_weather_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="python",
            args=["-m", "mcp_servers.weather_server"],
        ),
        timeout=60.0,
    )
)

_grid_data_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="python",
            args=["-m", "mcp_servers.grid_data_server"],
        ),
        timeout=60.0,
    )
)

_ml_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="python",
            args=["-m", "mcp_servers.ml_models_server"],
        ),
        timeout=60.0,
    )
)

_memory_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="python",
            args=["-m", "mcp_servers.memory_server"],
        ),
        timeout=60.0,
    )
)

_inner_agent = Agent(
    model=get_model("fast_agent"),
    name="renewables_forecast_skill",
    instruction=as_static_instruction(_INSTRUCTION),
    tools=[
        _weather_toolset,
        _grid_data_toolset,
        _ml_toolset,
        _memory_toolset,
    ],
    output_schema=RenewablesForecast,
    generate_content_config=genai_types.GenerateContentConfig(temperature=0.1),
)

RenewablesForecastSkill = AgentTool(agent=_inner_agent)
