"""
agents/orchestrator.py — OrchestratorAgent: Root Agent de GridGuard España.

Modelo: gemini-3.5-flash (fast_agent) · Temperature 0.1 · max_turns 20
Contexto: 1M tokens (permite ingestar semanas de histórico en una sola llamada)

Responsabilidades:
  1. Adquisición de datos de red (ree_poller → data_normalizer)
  2. Evaluación de riesgo (ML → risk_scorer)
  3. Escalación inteligente por capas según P(blackout):
       P < 0.10             → logging pasivo
       P ∈ [0.10, 0.30)    → AnomalyDetectorSkill
       P ∈ [0.30, 0.60)    → Anomaly + RenewablesForecastSkill + alerta WARNING
       P ≥ 0.60 (CRITICAL) → todo anterior + CrossBorderAgent + ReportAgent
  4. Persistencia en SQLite (store_snapshot, store_risk_score)
  5. Emisión de alertas formateadas

Sub-agents: report_agent, cross_border_agent
Skills (como AgentTools): AnomalyDetectorSkill, RenewablesForecastSkill
"""

from __future__ import annotations

import json
import logging
from typing import Any

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

from agents.anomaly_detector import AnomalyDetectorSkill
from agents.cross_border import cross_border_agent
from agents.renewables_forecast import RenewablesForecastSkill
from agents.report_agent import report_agent
from config.model_loader import get_model
from guardrails.hooks import register_hooks
from tools.alert_formatter import alert_formatter
from tools.data_normalizer import data_normalizer
from tools.ree_poller import ree_poller
from tools.risk_scorer import risk_scorer
from tools.stability_proxy_calculator import stability_proxy_calculator

log = logging.getLogger("gridguard.orchestrator")

_INSTRUCTION = """
Eres el OrchestratorAgent de GridGuard España, el operador autónomo de primer
nivel para la vigilancia de la red eléctrica peninsular española. Tu objetivo es
detectar situaciones de riesgo y escalar correctamente ANTES de que ocurran
problemas operacionales.

════════════════════════════════════════════════
PROTOCOLO DE MONITORIZACIÓN (seguir SIEMPRE este orden)
════════════════════════════════════════════════

PASO 1 — Adquisición de datos (OBLIGATORIO en cada ciclo):
  1a. Llama a `ree_poller` para obtener el snapshot de red en tiempo real.
  1b. Llama a `data_normalizer` con el resultado de 1a.
  1c. Llama a `stability_proxy_calculator` con el snapshot normalizado.
  1d. Llama a `get_blackout_probability` con los proxies calculados.
  1e. Llama a `risk_scorer` con el snapshot y el ml_score de 1d.
  1f. Llama a `store_snapshot` y `store_risk_score` para persistir los datos.

PASO 2 — Escalación según P(blackout_1h):

▶ P < 0.10 (VERDE):
  → Logging pasivo. Devuelve el RiskScore sin invocar Skills ni Sub-agents.
  → Mensaje: "Sistema en operación normal. P={P:.1%}, Risk={risk}/100."

▶ P ∈ [0.10, 0.30) (AMARILLO):
  → Invoca `AnomalyDetectorSkill` con el snapshot normalizado del paso 1b.
  → Registra el AnomalyReport en el log.
  → No emitir alerta pública (estado de vigilancia elevada).

▶ P ∈ [0.30, 0.60) (NARANJA):
  → Invoca `AnomalyDetectorSkill` si no se ha hecho.
  → Invoca `RenewablesForecastSkill` con el snapshot.
  → Llama a `alert_formatter` con severity determinada por RiskScore.level,
    scope="peninsular", y las anomalías del AnomalyReport.
  → Emite alerta WARNING en el log (con acciones recomendadas).

▶ P ≥ 0.60 (ROJO — EMERGENCIA):
  → Invoca `AnomalyDetectorSkill`.
  → Invoca `RenewablesForecastSkill`.
  → Invoca `CrossBorderAgent` para evaluar importación de emergencia.
  → Llama a `alert_formatter` con severity=CRITICAL, scope=nacional.
  → Invoca `ReportAgent` con report_type="emergency".
  → Emite alerta CRITICAL NACIONAL.

════════════════════════════════════════════════
REGLAS NO NEGOCIABLES (shift-left)
════════════════════════════════════════════════
- Nunca uses el LLM para calcular proxies de estabilidad — usa stability_proxy_calculator.
- Nunca uses el LLM para calcular el risk score — usa risk_scorer.
- Nunca hardcodees un riesgo sin consultar al ML primero (get_blackout_probability).
- Si una herramienta falla, regístralo y continúa con el resto del flujo.
- Si los modelos ML no están disponibles (error en get_blackout_probability),
  asume P=0.20 y activa AnomalyDetectorSkill de forma preventiva.

════════════════════════════════════════════════
FORMATO DE RESPUESTA FINAL
════════════════════════════════════════════════
Devuelve SIEMPRE un dict JSON con:
{
  "timestamp": "<ISO 8601 UTC>",
  "risk_score": <0-100>,
  "blackout_probability": <0.0-1.0>,
  "severity": "LOW|MEDIUM|HIGH|CRITICAL",
  "scope": "nacional|peninsular|zonal|local",
  "alert": { ... } o null,
  "anomaly_report": { ... } o null,
  "renewables_forecast": { ... } o null,
  "cross_border": { ... } o null,
  "actions_taken": ["lista de acciones ejecutadas"]
}
""".strip()

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

orchestrator = Agent(
    model=get_model("fast_agent"),
    name="gridguard_orchestrator",
    instruction=_INSTRUCTION,
    tools=[
        # Capa 1: Function Tools (Python puro, sin LLM)
        ree_poller,
        data_normalizer,
        stability_proxy_calculator,
        risk_scorer,
        alert_formatter,
        # Capa 2: MCP Tools (ML inference + memoria)
        _ml_toolset,
        _memory_toolset,
        # Capa 3: Skills (AgentTools — LLM con contexto específico)
        AnomalyDetectorSkill,
        RenewablesForecastSkill,
    ],
    sub_agents=[
        report_agent,
        cross_border_agent,
    ],
    temperature=0.1,
    max_turns=20,
)

# Registrar los 5 guardrails
register_hooks(orchestrator)


# ---------------------------------------------------------------------------
# API de ejecución
# ---------------------------------------------------------------------------

def run(region: str = "peninsular") -> dict[str, Any]:
    """Ejecuta un ciclo de monitorización en tiempo real.

    Devuelve el dict de resultado del Orchestrator.
    """
    session_service = InMemorySessionService()
    runner = Runner(
        agent=orchestrator,
        app_name="gridguard",
        session_service=session_service,
    )
    session = session_service.create_session(
        app_name="gridguard",
        user_id="gridguard-system",
    )
    result = runner.run(
        session_id=session.id,
        user_id="gridguard-system",
        new_message={"role": "user", "content": f"Ejecuta ciclo de monitorización para región: {region}"},
    )
    return _extract_result(result)


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
    session = session_service.create_session(
        app_name="gridguard_eval",
        user_id="eval-runner",
    )

    # Serializar el snapshot más reciente como contexto de entrada
    snapshot = scenario.get("snapshots", [{}])[-1]
    scenario_name = scenario.get("scenario_name", "unknown")
    message = (
        f"[EVAL] Escenario: {scenario_name}. "
        f"Analiza este snapshot de red y ejecuta el protocolo completo: "
        f"{json.dumps(snapshot)}"
    )

    result = runner.run(
        session_id=session.id,
        user_id="eval-runner",
        new_message={"role": "user", "content": message},
    )

    output = _extract_result(result)
    # Agregar trayectoria de tools para los tests de eval
    output["trajectory"] = _extract_trajectory(result)
    return output


def _extract_result(runner_result) -> dict:
    """Extrae el dict de resultado del runner de ADK."""
    try:
        last_message = runner_result.messages[-1] if runner_result.messages else {}
        content = last_message.get("content", "")
        if isinstance(content, str):
            stripped = content.strip()
            if stripped.startswith("{"):
                return json.loads(stripped)
        return {"raw": content}
    except Exception as exc:
        log.error("Error extrayendo resultado: %s", exc)
        return {"error": str(exc)}


def _extract_trajectory(runner_result) -> list[dict]:
    """Extrae la lista de tool calls del resultado del runner."""
    trajectory = []
    try:
        for msg in getattr(runner_result, "messages", []):
            tool_calls = msg.get("tool_calls") or []
            for tc in tool_calls:
                trajectory.append({
                    "tool": tc.get("name") or tc.get("function", {}).get("name", "unknown"),
                    "args": tc.get("args") or tc.get("function", {}).get("arguments", {}),
                })
    except Exception as exc:
        log.warning("Error extrayendo trayectoria: %s", exc)
    return trajectory


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    result = run()
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
