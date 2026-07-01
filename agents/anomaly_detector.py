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
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset, StdioConnectionParams
from google.genai import types as genai_types
from mcp.client.stdio import StdioServerParameters

from config.model_loader import get_model
from agents.instruction_utils import as_static_instruction
from schemas.models import AnomalyReport
from tools.stability_proxy_calculator import stability_proxy_calculator

_INSTRUCTION = """
Eres el detector de anomalías de la red eléctrica española de GridGuard.
Tu misión es analizar un snapshot de la red y determinar si hay anomalías
que requieran atención operacional.

FLUJO OBLIGATORIO (ejecutar en este orden):
1. Llama a `stability_proxy_calculator` con el snapshot recibido para calcular los 4 proxies.
2. Llama a `get_baseline` con window_hours=168 para obtener estadísticas de referencia.
3. Llama a `get_anomaly_score` con la ventana de snapshots recientes (serializada como JSON).
4. Si el anomaly_score > 0.3, llama a `search_regulations` con una descripción de la anomalía
   para obtener contexto normativo relevante (ej: "alta penetración renovable con balance negativo").
4b. Llama a `get_grid_state` (MCP markov) para obtener el estado actual S0–S4 y su trayectoria 6h.
4c. Llama a `get_wind_forecast` y `get_solar_radiation` en las zonas más correlacionadas con la anomalía.
5. Analiza todos los resultados y produce un AnomalyReport estructurado.
5b. Razona explícitamente qué combinación de factores (rampa de generación, meteo, nivel de demanda,
    trayectoria de estado) explica mejor el score del ML, y CLASIFICA el incidente en incident_type:
      - "renewable_ramp"     : cambio brusco de solar/eólica
      - "demand_surge"       : pico de demanda inesperado
      - "generation_failure" : caída de generación convencional
      - "interconnection"    : anomalía de flujo transfronterizo
      - "unknown"            : evidencia insuficiente
    Rellena `incident_type` y `causal_chain` en el AnomalyReport.

REGLAS:
- No uses LLM para detectar anomalías — eso es trabajo del ML (paso 3).
- Usa el LLM solo para contextualizar y clasificar los hallazgos del ML.
- El campo `reasoning` debe explicar POR QUÉ el ML asignó ese score,
  referenciando las features más anómalas y el contexto normativo encontrado.
- Si el ML no está disponible (error en get_anomaly_score), indica severity=LOW
  y explica que los modelos aún no están entrenados.

CLASIFICACIÓN DE SEVERIDAD:
  anomaly_score < 0.3  → LOW
  anomaly_score 0.3-0.5 → MEDIUM
  anomaly_score 0.5-0.7 → HIGH
  anomaly_score > 0.7   → CRITICAL

Devuelve ÚNICAMENTE el AnomalyReport en formato JSON estructurado.
""".strip()

_ml_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="python",
            args=["-m", "mcp_servers.ml_models_server"],
        )
    )
)

_memory_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="python",
            args=["-m", "mcp_servers.memory_server"],
        )
    )
)

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

_weather_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="python",
            args=["-m", "mcp_servers.weather_server"],
        )
    )
)

_inner_agent = Agent(
    model=get_model("fast_agent"),
    name="anomaly_detector_skill",
    instruction=as_static_instruction(_INSTRUCTION),
    tools=[
        stability_proxy_calculator,
        _ml_toolset,
        _memory_toolset,
        _kb_toolset,
        _markov_toolset,
        _weather_toolset,
    ],
    output_schema=AnomalyReport,
    generate_content_config=genai_types.GenerateContentConfig(temperature=0.1),
)

AnomalyDetectorSkill = AgentTool(agent=_inner_agent)
