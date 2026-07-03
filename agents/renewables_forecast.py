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
from google.genai import types as genai_types

from agents.mcp_toolsets import GRID_DATA_TOOLSET, MEMORY_TOOLSET, ML_TOOLSET, WEATHER_TOOLSET
from config.model_loader import get_model
from agents.instruction_utils import as_static_instruction, load_prompt
from schemas.models import RenewablesForecast

_INSTRUCTION = load_prompt("renewables_forecast")

_inner_agent = Agent(
    model=get_model("fast_agent"),
    name="renewables_forecast_skill",
    instruction=as_static_instruction(_INSTRUCTION),
    tools=[
        WEATHER_TOOLSET,
        GRID_DATA_TOOLSET,
        ML_TOOLSET,
        MEMORY_TOOLSET,
    ],
    output_schema=RenewablesForecast,
    generate_content_config=genai_types.GenerateContentConfig(temperature=0.1),
)

RenewablesForecastSkill = AgentTool(agent=_inner_agent)
