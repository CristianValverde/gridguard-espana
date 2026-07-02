"""
agents/regulatory_compliance.py -- Skill: RegulatoryComplianceSkill (AgentTool).

Tipo: AgentTool (sincrono, sin estado). Modelo: fast_agent, temperature 0.1.
Trigger (orquestador): P>=0.30 O estado de Markov >= S2.
Determina que procedimientos operativos REE/ENTSO-E aplican al estado actual de la red.
"""

from __future__ import annotations

from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset, StdioConnectionParams
from google.genai import types as genai_types
from mcp.client.stdio import StdioServerParameters

from config.model_loader import get_model
from agents.instruction_utils import as_static_instruction, load_prompt
from schemas.models import ComplianceReport

_INSTRUCTION = load_prompt("regulatory_compliance")

_kb_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="python",
            args=["-m", "mcp_servers.knowledge_base_server"],
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

_inner_agent = Agent(
    model=get_model("fast_agent"),
    name="regulatory_compliance_skill",
    instruction=as_static_instruction(_INSTRUCTION),
    tools=[_kb_toolset, _ml_toolset],
    output_schema=ComplianceReport,
    generate_content_config=genai_types.GenerateContentConfig(temperature=0.1),
)

RegulatoryComplianceSkill = AgentTool(agent=_inner_agent)
