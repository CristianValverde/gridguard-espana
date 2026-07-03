"""
agents/regulatory_compliance.py -- Skill: RegulatoryComplianceSkill (AgentTool).

Tipo: AgentTool (sincrono, sin estado). Modelo: fast_agent, temperature 0.1.
Trigger (orquestador): P>=0.30 O estado de Markov >= S2.
Determina que procedimientos operativos REE/ENTSO-E aplican al estado actual de la red.
"""

from __future__ import annotations

from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from google.genai import types as genai_types

from agents.mcp_toolsets import KB_TOOLSET, ML_TOOLSET
from config.model_loader import get_model
from agents.instruction_utils import as_static_instruction, load_prompt
from schemas.models import ComplianceReport

_INSTRUCTION = load_prompt("regulatory_compliance")

_inner_agent = Agent(
    model=get_model("fast_agent"),
    name="regulatory_compliance_skill",
    instruction=as_static_instruction(_INSTRUCTION),
    tools=[KB_TOOLSET, ML_TOOLSET],
    output_schema=ComplianceReport,
    generate_content_config=genai_types.GenerateContentConfig(temperature=0.1),
)

RegulatoryComplianceSkill = AgentTool(agent=_inner_agent)
