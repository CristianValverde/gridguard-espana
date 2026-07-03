"""
agents/anomaly_detector.py — Skill: AnomalyDetectorSkill (AgentTool).

Tipo: AgentTool (síncrono, sin estado entre llamadas).
Modelo: fast_agent · Temperature 0.1
Flujo: ML detecta → RAG contextualiza → Markov + Meteo razonan causalidad → LLM clasifica.

Tools disponibles:
  - get_anomaly_score      (MCP: ml-models)
  - stability_proxy_calculator  (Function Tool)
  - get_baseline           (MCP: memory)
  - search_regulations     (MCP: knowledge-base)
  - get_grid_state         (MCP: markov)
  - get_wind_forecast      (MCP: weather)
  - get_solar_radiation    (MCP: weather)

Output schema: AnomalyReport
"""

from __future__ import annotations

from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from google.genai import types as genai_types

from agents.mcp_toolsets import KB_TOOLSET, MEMORY_TOOLSET, ML_TOOLSET, WEATHER_TOOLSET
from config.model_loader import get_model
from agents.instruction_utils import as_static_instruction, load_prompt
from schemas.models import AnomalyReport
from tools.stability_proxy_calculator import stability_proxy_calculator

_INSTRUCTION = load_prompt("anomaly_detector")

_inner_agent = Agent(
    model=get_model("fast_agent"),
    name="anomaly_detector_skill",
    instruction=as_static_instruction(_INSTRUCTION),
    tools=[
        stability_proxy_calculator,
        ML_TOOLSET,
        MEMORY_TOOLSET,
        KB_TOOLSET,
        WEATHER_TOOLSET,
    ],
    output_schema=AnomalyReport,
    generate_content_config=genai_types.GenerateContentConfig(temperature=0.1),
)

AnomalyDetectorSkill = AgentTool(agent=_inner_agent)
