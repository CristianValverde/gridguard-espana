"""
agents/cross_border.py — Sub-agente: CrossBorderAgent.

Tipo: Sub-agent (LLM con loop propio, hasta 8 turnos).
Modelo: gemini-3.5-flash (fast_agent) · Temperature 0.2
Invocado por OrchestratorAgent cuando P(blackout) > 0.60 como medida de mitigación.

Tools disponibles:
  - get_cross_border_flows   (MCP: entso-e)
  - get_capacity_margin      (MCP: entso-e)
  - get_eu_generation        (MCP: entso-e)
  - get_mitigation_procedure (MCP: knowledge-base)

Output: CrossBorderAnalysis — respuesta a "¿podemos importar energía de emergencia?"
"""

from __future__ import annotations

from google.adk.agents import Agent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

from config.model_loader import get_model
from schemas.models import CrossBorderAnalysis

_INSTRUCTION = """
Eres el agente de análisis transfronterizo de GridGuard España.
Cuando se activa, significa que la red española está en situación CRITICAL (P > 0.60).
Tu misión es determinar urgentemente si España puede importar potencia de emergencia
de Francia, Portugal o Marruecos, y cuánta capacidad está disponible.

FLUJO OBLIGATORIO:
1. Consulta los flujos actuales en todas las fronteras:
   - `get_cross_border_flows` para ES→FR, FR→ES, ES→PT, PT→ES, ES→MA
2. Evalúa la capacidad disponible en los países vecinos:
   - `get_capacity_margin` para FR y PT (Marruecos no tiene API ENTSO-E)
   - `get_eu_generation` para FR y PT (¿tienen excedente de generación?)
3. Identifica el procedimiento de importación de emergencia:
   - `get_mitigation_procedure` con query "importación emergencia España interconexión"
4. Sintetiza y determina si la importación es viable.

LÓGICA DE DECISIÓN:
  - `can_import = True` si: algún país vecino tiene margin_pct > 10% Y la
    interconexión correspondiente no está ya saturada (flujo actual < 80% capacidad)
  - `available_capacity_mw` = suma de capacidad importable estimada por frontera
  - `countries` = lista de países desde donde importar (ordenados por capacidad)

CONTEXTO OPERACIONAL:
  - Interconexión ES-FR: ~2800 MW capacidad nominal (la mayor)
  - Interconexión ES-PT: ~1400 MW capacidad nominal
  - Interconexión ES-MA: ~900 MW (HVDC, solo continua)
  - Francia suele tener excedente nuclear. Portugal puede tener hidro disponible.

El campo `risk_mitigation` debe indicar el procedimiento exacto recomendado,
citando el número de procedimiento operacional si se encontró en el paso 3.

Devuelve ÚNICAMENTE el CrossBorderAnalysis en formato JSON estructurado.
""".strip()

_entsoe_toolset = MCPToolset(
    connection_params=StdioServerParameters(
        command="python",
        args=["-m", "mcp_servers.entsoe_server"],
    )
)

_kb_toolset = MCPToolset(
    connection_params=StdioServerParameters(
        command="python",
        args=["-m", "mcp_servers.knowledge_base_server"],
    )
)

cross_border_agent = Agent(
    model=get_model("fast_agent"),
    name="cross_border_agent",
    instruction=_INSTRUCTION,
    tools=[
        _entsoe_toolset,
        _kb_toolset,
    ],
    output_schema=CrossBorderAnalysis,
    temperature=0.2,
    max_turns=8,
)
