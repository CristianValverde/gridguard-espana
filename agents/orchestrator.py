"""
agents/orchestrator.py -- OrchestratorAgent: Root Agent de GridGuard Espana.

Modelo: gemini-3.5-flash (fast_agent) - Temperature 0.1 (via GenerateContentConfig)
Contexto: 1M tokens (permite ingestar semanas de historico en una sola llamada)

ADK 2.3.0 migration notes:
- MCPToolset -> McpToolset; los toolsets stdio son singletons compartidos en
  agents/mcp_toolsets.py (un subproceso por server, no por agente x server).
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
  5. Persistencia del risk score en SQLite (store_risk_score)
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
from google.genai import types as genai_types

from agents.anomaly_detector import AnomalyDetectorSkill
from agents.cross_border import cross_border_agent
from agents.mcp_toolsets import MEMORY_TOOLSET, ML_TOOLSET
from agents.regulatory_compliance import RegulatoryComplianceSkill
from agents.renewables_forecast import RenewablesForecastSkill
from agents.report_agent import report_agent
from config.model_loader import get_model
from agents.instruction_utils import as_static_instruction, load_prompt
from guardrails.hooks import register_hooks
from tools.alert_formatter import alert_formatter
from tools.data_normalizer import data_normalizer
from tools.ree_poller import ree_poller
from tools.escalation_planner import escalation_planner
from tools.risk_scorer import risk_scorer
from tools.stability_proxy_calculator import stability_proxy_calculator

log = logging.getLogger("gridguard.orchestrator")

_INSTRUCTION = load_prompt("orchestrator")

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
        escalation_planner,
        alert_formatter,
        # Capa 2: MCP Tools (ML inference + Markov + memoria)
        ML_TOOLSET,
        MEMORY_TOOLSET,
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

def _consume_events_safely(result) -> list:
    """Consume el generador de Events del Runner, tolerando que un sub-agente
    (AgentTool con output_schema estricto) lance una excepcion no capturada
    por ADK 2.3.0 (ej. pydantic.ValidationError en agent_tool.py::run_async
    cuando la skill responde texto libre en vez de JSON). Sin esto, UN turno
    mal formateado aborta TODO el ciclo de monitorizacion sin producir ningun
    resultado; con esto, se degrada a "esa skill no produjo output valido" y
    el flujo continua (incluido escalation_enforcer, que puede reparar el hueco).
    """
    events: list = []
    try:
        for event in result:
            events.append(event)
    except Exception as exc:
        log.warning(
            "Runner.run() interrumpido por excepcion no capturada (probable "
            "output_schema invalido de una skill): %s. Continuando con %d "
            "eventos ya capturados.", exc, len(events),
        )
    return events


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
    events = _consume_events_safely(runner.run(
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
    from agents.escalation_enforcer import enforce_escalation_plan  # import diferido: evita ciclo con escalation_enforcer, que importa de este modulo

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
    events = _consume_events_safely(result)
    output = _extract_result(events)
    # Agregar trayectoria de tools para los tests de eval
    output["trajectory"] = _extract_trajectory(events)
    # Enforcer determinista: repara por codigo lo que el LLM no ejecuto de lo
    # que escalation_planner marco como obligatorio (ver
    # docs/superpowers/specs/2026-07-04-escalation-enforcer-design.md).
    output = enforce_escalation_plan(output, output["trajectory"], snapshot)
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
