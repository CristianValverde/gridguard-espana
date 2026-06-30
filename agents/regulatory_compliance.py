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
from schemas.models import ComplianceReport

_INSTRUCTION = """
Eres el agente de cumplimiento normativo de GridGuard España. Dado el estado actual de la
red, determinas que procedimientos operativos (PO) de REE/ENTSO-E aplican y si la situación
dispara acciones obligatorias.

FLUJO:
1. Recibes un snapshot (renewables_pct, demand_mw, balance_mw, grid_state, anomalías).
2. Llama a `get_grid_state` para confirmar el estado de Markov y su trayectoria.
3. Llama a `search_regulations` con una query construida desde el estado actual, p.ej.:
     - renovables altas  -> "procedimiento operativo frecuencia inercia renovables"
     - reserva activando  -> "reserva secundaria activación umbral"
     - estado >= S3        -> "importación emergencia interconexión"
4. Para cada regulación encontrada, evalúa si el estado actual la dispara.
5. Devuelve un ComplianceReport.

REGLAS:
- NUNCA calcules umbrales numéricos tú mismo -- vienen en el snapshot.
- Cita SOLO procedimientos devueltos por `search_regulations`; NUNCA inventes IDs de PO.
- Si no hay regulaciones aplicables, devuelve compliance_status="compliant".

Devuelve ÚNICAMENTE el ComplianceReport en JSON estructurado.
""".strip()

_kb_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="python",
            args=["-m", "mcp_servers.knowledge_base_server"],
        )
    )
)

_markov_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="python",
            args=["-m", "mcp_servers.markov_server"],
        )
    )
)

_inner_agent = Agent(
    model=get_model("fast_agent"),
    name="regulatory_compliance_skill",
    instruction=_INSTRUCTION,
    tools=[_kb_toolset, _markov_toolset],
    output_schema=ComplianceReport,
    generate_content_config=genai_types.GenerateContentConfig(temperature=0.1),
)

RegulatoryComplianceSkill = AgentTool(agent=_inner_agent)
