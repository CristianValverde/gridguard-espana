"""
agents/orchestrator.py -- OrchestratorAgent: Root Agent de GridGuard Espana.

Modelo: gemini-3.5-flash (fast_agent) - Temperature 0.1 (via GenerateContentConfig)
Contexto: 1M tokens (permite ingestar semanas de historico en una sola llamada)

ADK 2.3.0 migration notes:
- MCPToolset -> McpToolset; StdioServerParameters wrapped in StdioConnectionParams.
- temperature -> generate_content_config=genai_types.GenerateContentConfig(temperature=...).
- max_turns removed (not permitted in ADK 2.3.0; turn-limiting moves to the runner).
- register_hooks: ADK 2.3.0 exposes before_tool_callback / after_tool_callback /
  before_model_callback / after_model_callback / after_agent_callback as proper
  pydantic fields on LlmAgent. Direct post-construction assignment works
  (model_config allows arbitrary_types_allowed=True). The existing hook functions
  in guardrails/hooks.py have signatures that differ from the ADK 2.3.0 protocol
  (ADK 2.3.0 passes BaseTool+dict+Context; hooks.py uses simpler args). Hooks are
  registered as-is so guardrail wiring is preserved; runtime behaviour under ADK
  2.3.0 will need adaptation in a follow-up task (callback signature mismatch).

Responsabilidades:
  1. Adquisicion de datos de red (ree_poller -> data_normalizer)
  2. Evaluacion de riesgo (ML -> risk_scorer)
  3. Adquisicion de estado de Markov (get_grid_state) + combinacion con P(blackout)
  4. Escalacion inteligente por capas segun P(blackout) y estado de Markov:
       P < 0.10             -> logging pasivo (override si Markov >= S2)
       P in [0.10, 0.30)   -> AnomalyDetectorSkill (+ CrossBorder assessment si S1)
       P in [0.30, 0.60)   -> Anomaly + Renewables + RegulatoryCompliance + CrossBorder assessment
       P >= 0.60 (CRITICAL) -> todo anterior + CrossBorderAgent + ReportAgent
  5. Persistencia en SQLite (store_snapshot, store_risk_score)
  6. Emision de alertas formateadas

Sub-agents: report_agent, cross_border_agent
Skills (como AgentTools): AnomalyDetectorSkill, RenewablesForecastSkill, RegulatoryComplianceSkill
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset, StdioConnectionParams
from google.genai import types as genai_types
from mcp.client.stdio import StdioServerParameters

from agents.anomaly_detector import AnomalyDetectorSkill
from agents.cross_border import cross_border_agent
from agents.regulatory_compliance import RegulatoryComplianceSkill
from agents.renewables_forecast import RenewablesForecastSkill
from agents.report_agent import report_agent
from config.model_loader import get_model
from agents.instruction_utils import as_static_instruction
from guardrails.hooks import register_hooks
from tools.alert_formatter import alert_formatter
from tools.data_normalizer import data_normalizer
from tools.ree_poller import ree_poller
from tools.risk_scorer import risk_scorer
from tools.stability_proxy_calculator import stability_proxy_calculator

log = logging.getLogger("gridguard.orchestrator")

_INSTRUCTION = """
Eres el OrchestratorAgent de GridGuard Espana, el operador autonomo de primer
nivel para la vigilancia de la red electrica peninsular espanola. Tu objetivo es
detectar situaciones de riesgo y escalar correctamente ANTES de que ocurran
problemas operacionales.

================================================
PROTOCOLO DE MONITORIZACION (seguir SIEMPRE este orden)
================================================

PASO 1 -- Adquisicion de datos (OBLIGATORIO en cada ciclo):
  1a. Llama a `ree_poller` para obtener el snapshot de red en tiempo real.
  1b. Llama a `data_normalizer` con el resultado de 1a.
  1c. Llama a `stability_proxy_calculator` con el snapshot normalizado.
  1d. [NUEVO] Llama a `get_grid_state` (MCP markov) para obtener el estado actual S0-S4
      y la trayectoria de las ultimas 6h.
  1e. Llama a `get_blackout_probability_ensemble` para obtener P(blackout). Este tool
      fusiona LSTM (load-bearing) + Isolation Forest + clasificador usando las features
      de serve (paridad train/serve, ventana 168h de la DB). Usa su `blackout_probability`.
  1f. Llama a `risk_scorer` con el snapshot y el ml_score de 1e (el blackout_probability).
  1g. Llama a `store_snapshot` y `store_risk_score` para persistir los datos.

PASO 2 -- Escalacion segun P(blackout_1h) Y estado de Markov:

La escalacion usa AHORA tanto P(blackout) COMO el estado de Markov:
  - Si estado Markov >= S2 Y P < 0.10: eleva a AMARILLO (el estado manda cuando
    el ML duda pero la trayectoria se degrada claramente).
  - Si estado Markov = S1 Y P in [0.10, 0.30): activa CrossBorderAgent en modo
    assessment ademas del flujo normal AMARILLO.

Tramos de escalacion:

[VERDE] P < 0.10 (y Markov <= S1):
  -> Logging pasivo. Devuelve el RiskScore sin invocar Skills ni Sub-agents.
  -> Mensaje: "Sistema en operacion normal. P={P:.1%}, Risk={risk}/100."
  -> EXCEPCION: si Markov >= S2, elevar a AMARILLO aunque P < 0.10.

[AMARILLO] P in [0.10, 0.30) (o override por Markov >= S2 con P < 0.10):
  -> Invoca `AnomalyDetectorSkill` con el snapshot normalizado del paso 1b.
  -> Registra el AnomalyReport en el log.
  -> Si estado Markov = S1 Y P in [0.10, 0.30): activa CrossBorderAgent
     con mode="assessment".
  -> No emitir alerta publica (estado de vigilancia elevada).

[NARANJA] P in [0.30, 0.60) -- [ACTUALIZADO]:
  -> Invoca `AnomalyDetectorSkill` si no se ha hecho.
  -> Invoca `RenewablesForecastSkill` con el snapshot.
  -> [NUEVO] Invoca `RegulatoryComplianceSkill` con el estado actual de la red.
  -> [NUEVO] Invoca CrossBorderAgent con mode="assessment".
  -> Llama a `alert_formatter` con las 4 entradas anteriores: severity del
     RiskScore.level, scope="peninsular", anomalias del AnomalyReport,
     y capacidad de importacion disponible.
  -> Emite alerta WARNING incluyendo la capacidad de importacion disponible.

[ROJO -- EMERGENCIA] P >= 0.60:
  -> Invoca `AnomalyDetectorSkill`.
  -> Invoca `RenewablesForecastSkill`.
  -> Invoca `RegulatoryComplianceSkill`.
  -> Invoca `CrossBorderAgent` para evaluar importacion de emergencia.
  -> Llama a `alert_formatter` con severity=CRITICAL, scope=nacional.
  -> Invoca `ReportAgent` con report_type="emergency".
  -> Emite alerta CRITICAL NACIONAL.

================================================
REGLAS NO NEGOCIABLES (shift-left)
================================================
- Nunca uses el LLM para calcular proxies de estabilidad -- usa stability_proxy_calculator.
- Nunca uses el LLM para calcular el risk score -- usa risk_scorer.
- Nunca hardcodees un riesgo sin consultar al ML primero (get_blackout_probability_ensemble).
- Si una herramienta falla, registralo y continua con el resto del flujo.
- Si los modelos ML no estan disponibles (error o P=0 en get_blackout_probability_ensemble),
  asume P=0.20 y activa AnomalyDetectorSkill de forma preventiva.
- Siempre consulta get_grid_state antes de aplicar los umbrales de escalacion.

================================================
FORMATO DE RESPUESTA FINAL
================================================
Devuelve SIEMPRE un dict JSON con:
{
  "timestamp": "<ISO 8601 UTC>",
  "risk_score": <0-100>,
  "blackout_probability": <0.0-1.0>,
  "markov_state": "<S0|S1|S2|S3|S4>",
  "severity": "LOW|MEDIUM|HIGH|CRITICAL",
  "scope": "nacional|peninsular|zonal|local",
  "alert": { ... } o null,
  "anomaly_report": { ... } o null,
  "renewables_forecast": { ... } o null,
  "compliance_report": { ... } o null,
  "cross_border": { ... } o null,
  "actions_taken": ["lista de acciones ejecutadas"]
}
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

_markov_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="python",
            args=["-m", "mcp_servers.markov_server"],
        )
    )
)

orchestrator = Agent(
    model=get_model("fast_agent"),
    name="gridguard_orchestrator",
    instruction=as_static_instruction(_INSTRUCTION),
    tools=[
        # Capa 1: Function Tools (Python puro, sin LLM)
        ree_poller,
        data_normalizer,
        stability_proxy_calculator,
        risk_scorer,
        alert_formatter,
        # Capa 2: MCP Tools (ML inference + memoria + Markov)
        _ml_toolset,
        _memory_toolset,
        _markov_toolset,
        # Capa 3: Skills (AgentTools -- LLM con contexto especifico)
        AnomalyDetectorSkill,
        RenewablesForecastSkill,
        RegulatoryComplianceSkill,
    ],
    sub_agents=[
        report_agent,
        cross_border_agent,
    ],
    generate_content_config=genai_types.GenerateContentConfig(temperature=0.1),
)

# Registrar los 5 guardrails.
# NOTA ADK 2.3.0: Las firmas de callback en guardrails/hooks.py son
# (tool_name, tool_args, agent_name) para before_tool_callback y
# (tool_name, output) para after_tool_callback. ADK 2.3.0 espera
# (BaseTool, dict, Context) y (BaseTool, dict, Context, dict) respectivamente.
# La asignacion directa funciona (pydantic no valida el callable en tiempo de
# asignacion bajo arbitrary_types_allowed=True), por lo que el wiring de
# guardrails queda registrado. La adaptacion de firmas se marca como TECH DEBT
# para un follow-up task de migracion de callbacks.
register_hooks(orchestrator)


# ---------------------------------------------------------------------------
# API de ejecucion
# ---------------------------------------------------------------------------

def run(region: str = "peninsular") -> dict[str, Any]:
    """Ejecuta un ciclo de monitorizacion en tiempo real.

    Devuelve el dict de resultado del Orchestrator.
    """
    session_service = InMemorySessionService()
    runner = Runner(
        agent=orchestrator,
        app_name="gridguard",
        session_service=session_service,
    )
    # ADK 2.3.0: create_session is a coroutine — must be awaited via asyncio.run.
    session = asyncio.run(
        session_service.create_session(
            app_name="gridguard",
            user_id="gridguard-system",
        )
    )
    message = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=f"Ejecuta ciclo de monitorizacion para region: {region}")],
    )
    # ADK 2.3.0: runner.run() returns a Generator[Event, None, None] — consume it.
    events = list(runner.run(
        session_id=session.id,
        user_id="gridguard-system",
        new_message=message,
    ))
    return _extract_result(events)


def run_scenario(scenario: dict) -> dict[str, Any]:
    """Ejecuta el Orchestrator sobre un escenario de datos (para eval/testing).

    Args:
        scenario: Dict con snapshots pre-cargados y metadata del escenario.

    Returns:
        Dict con resultado + trayectoria de tools llamadas (para asserts de eval).
    """
    session_service = InMemorySessionService()
    runner = Runner(
        agent=orchestrator,
        app_name="gridguard_eval",
        session_service=session_service,
    )
    # ADK 2.3.0: create_session is a coroutine — must be awaited via asyncio.run.
    session = asyncio.run(
        session_service.create_session(
            app_name="gridguard_eval",
            user_id="eval-runner",
        )
    )

    # Serializar el snapshot mas reciente como contexto de entrada
    snapshot = scenario.get("snapshots", [{}])[-1]
    scenario_name = scenario.get("scenario_name", "unknown")
    message_text = (
        f"[EVAL] Escenario: {scenario_name}. "
        f"Analiza este snapshot de red y ejecuta el protocolo completo: "
        f"{json.dumps(snapshot)}"
    )
    message = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=message_text)],
    )

    result = runner.run(
        session_id=session.id,
        user_id="eval-runner",
        new_message=message,
    )

    # ADK 2.3.0: runner.run() returns a Generator[Event, None, None] — consume it.
    events = list(result)
    output = _extract_result(events)
    # Agregar trayectoria de tools para los tests de eval
    output["trajectory"] = _extract_trajectory(events)
    return output


def _extract_result(events: list) -> dict:
    """Extrae el dict de resultado de la lista de Events de ADK 2.3.0.

    Busca el último evento de respuesta final del agente orquestador y
    extrae el JSON del texto de su contenido.
    """
    try:
        # Iterar en reversa para encontrar la última respuesta del agente
        for event in reversed(events):
            if not event.is_final_response():
                continue
            content = getattr(event, "content", None)
            if content is None:
                continue
            parts = getattr(content, "parts", None) or []
            for part in parts:
                text = getattr(part, "text", None) or ""
                stripped = text.strip()
                if stripped.startswith("{"):
                    return json.loads(stripped)
                elif stripped:
                    return {"raw": stripped}
        return {"raw": ""}
    except Exception as exc:
        log.error("Error extrayendo resultado: %s", exc)
        return {"error": str(exc)}


def _extract_trajectory(events: list) -> list[dict]:
    """Extrae la lista de tool calls de la lista de Events de ADK 2.3.0."""
    trajectory = []
    try:
        for event in events:
            content = getattr(event, "content", None)
            if content is None:
                continue
            parts = getattr(content, "parts", None) or []
            for part in parts:
                fc = getattr(part, "function_call", None)
                if fc is not None:
                    trajectory.append({
                        "tool": getattr(fc, "name", "unknown"),
                        "args": getattr(fc, "args", {}),
                    })
    except Exception as exc:
        log.warning("Error extrayendo trayectoria: %s", exc)
    return trajectory


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    result = run()
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
