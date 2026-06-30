"""
agents/renewables_forecast.py — Skill: RenewablesForecastSkill (AgentTool).

Tipo: AgentTool (síncrono, sin estado entre llamadas).
Modelo: gemini-3.5-flash (fast_agent) · Temperature 0.1
Flujo: Previsiones meteorológicas + ML demand forecast → análisis integrado.

Tools disponibles:
  - get_wind_forecast    (MCP: weather)
  - get_solar_radiation  (MCP: weather)
  - get_generation_mix   (MCP: ree-grid)
  - get_load_forecast    (MCP: ml-models)
  - get_baseline         (MCP: memory)

Output schema: RenewablesForecast
"""

from __future__ import annotations

from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

from config.model_loader import get_model
from schemas.models import RenewablesForecast

_INSTRUCTION = """
Eres el analista de previsión de renovables de GridGuard España.
Tu misión es evaluar el riesgo operacional derivado de la generación renovable
en las próximas 6 horas y determinar la ventana crítica de máximo riesgo.

FLUJO OBLIGATORIO:
1. Llama a `get_load_forecast` con horizon_hours=6 para obtener la previsión de demanda ML.
2. Llama a `get_baseline` con window_hours=168 para contexto histórico.
3. Llama a `get_wind_forecast` para las zonas clave (norte, interior, galicia).
4. Llama a `get_solar_radiation` para las zonas clave (sur, levante).
5. Llama a `get_generation_mix` para el estado actual de generación.
6. Sintetiza los resultados e identifica la ventana de máximo riesgo.

CRITERIOS DE RIESGO DE RENOVABLES:
  expected_generation_pct < 20%  → LOW     (baja generación, fácil gestión)
  expected_generation_pct 20-40% → MEDIUM  (gestión estándar)
  expected_generation_pct 40-60% → HIGH    (elevada penetración, vigilar inercia)
  expected_generation_pct > 60%  → CRITICAL (muy alta penetración, riesgo de frecuencia)

VENTANA CRÍTICA: El período de 2h con mayor ratio renovables/demanda según las
previsiones. Si el viento es fuerte por la noche (demanda baja) o la solar en
el mediodía de un domingo (demanda mínima), el riesgo se dispara.

El campo `reasoning` debe explicar qué combinación de factores crea el riesgo
(por qué esa ventana, qué zonas contribuyen más, cómo se compara con el baseline).

Devuelve ÚNICAMENTE el RenewablesForecast en formato JSON estructurado.
""".strip()

_weather_toolset = MCPToolset(
    connection_params=StdioServerParameters(
        command="python",
        args=["-m", "mcp_servers.weather_server"],
    )
)

_ree_toolset = MCPToolset(
    connection_params=StdioServerParameters(
        command="python",
        args=["-m", "mcp_servers.ree_grid_server"],
    )
)

_ml_toolset = MCPToolset(
    connection_params=StdioServerParameters(
        command="python",
        args=["-m", "mcp_servers.ml_models_server"],
    )
)

_memory_toolset = MCPToolset(
    connection_params=StdioServerParameters(
        command="python",
        args=["-m", "mcp_servers.memory_server"],
    )
)

_inner_agent = Agent(
    model=get_model("fast_agent"),
    name="renewables_forecast_skill",
    instruction=_INSTRUCTION,
    tools=[
        _weather_toolset,
        _ree_toolset,
        _ml_toolset,
        _memory_toolset,
    ],
    output_schema=RenewablesForecast,
    temperature=0.1,
)

RenewablesForecastSkill = AgentTool(agent=_inner_agent)
