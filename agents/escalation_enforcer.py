"""agents/escalation_enforcer.py — repara por codigo la infra-escalacion.

El LLM del orchestrator no obedece consistentemente el resultado VINCULANTE
de escalation_planner (verificado en la corrida live 2026-07-04): en ROJO se
detiene antes de invocar todo lo obligatorio. Este modulo recalcula el plan
determinista con los valores YA presentes en el resultado final, detecta que
falta contra la trayectoria real, y lo invoca por codigo -- sin pasar por el
LLM del orchestrator, sin tocar el prompt.

Alcance (spec 2026-07-04-escalation-enforcer-design.md): repara SOLO
infra-escalacion (huecos en NARANJA/ROJO). NO deshace sobre-escalacion (VERDE
disparando de mas) -- eso es un residual de no-determinismo LLM aceptado, no
un riesgo de seguridad.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from agents.orchestrator import _extract_result, _extract_trajectory
from tools.alert_formatter import alert_formatter
from tools.escalation_planner import escalation_planner

_SUB_AGENT_NAMES = {"cross_border_agent", "report_agent"}


def _run_agent_once(agent, message_text: str) -> dict[str, Any]:
    """Ejecuta `agent` de forma aislada (runner propio) y devuelve
    {"output": dict, "trajectory": list[{"tool": str, ...}]}.

    Mismo patron que eval/agents/_harness.py::run_agent_live, duplicado
    deliberadamente para no crear una dependencia inversa agents/ -> eval/.
    """
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="gridguard_enforcer", session_service=session_service)
    session = asyncio.run(
        session_service.create_session(app_name="gridguard_enforcer", user_id="enforcer")
    )
    message = genai_types.Content(role="user", parts=[genai_types.Part(text=message_text)])
    events = list(runner.run(session_id=session.id, user_id="enforcer", new_message=message))
    return {"output": _extract_result(events), "trajectory": _extract_trajectory(events)}


def _markov_state_int(label: str) -> int:
    try:
        return int(str(label).lstrip("S"))
    except (ValueError, TypeError):
        return 0


def _force(agent, name: str, message: str, result: dict, out_trajectory: list[dict]) -> dict:
    """Invoca `agent` forzado, fusiona su trayectoria (con bubbling si aplica)
    y devuelve su output. No modifica `result` directamente (lo hace el llamador)."""
    forced = _run_agent_once(agent, message)
    out_trajectory.append({"tool": name, "args": {"forced": True}})
    if name in _SUB_AGENT_NAMES:
        # Bubbling: los sub_agents ya burbujean sus tool-calls internas en el
        # flujo normal (transferencia ADK); replicarlo aqui para que asserts
        # como el de RAG (find_similar_incident dentro de report_agent) pasen.
        out_trajectory.extend(forced["trajectory"])
    result["actions_taken"].append(f"{name} (forzado por enforcer)")
    return forced["output"]


def enforce_escalation_plan(result: dict, trajectory: list[dict], snapshot: dict) -> dict:
    """Garantiza por codigo que todo lo que escalation_planner marca como
    obligatorio se ejecuto, invocando lo que falte de forma aislada.

    Args:
        result: dict de resultado del orchestrator (FORMATO DE RESPUESTA FINAL).
        trajectory: lista de {"tool": str, "args": dict} ya capturada del run del LLM.
        snapshot: GridSnapshot (dict) del ciclo actual, para construir los mensajes
            de las invocaciones forzadas.

    Returns:
        `result` mutado (mismas claves), con `result["trajectory"]` actualizado.
    """
    p = result.get("blackout_probability", 0.0) or 0.0
    state = _markov_state_int(result.get("markov_state", "S0"))
    plan = escalation_planner(p, state)

    trajectory = list(trajectory)  # copia local, no mutar el arg original
    tool_names = {step["tool"] for step in trajectory}

    context_json = json.dumps({
        "snapshot": snapshot,
        "risk_score": result.get("risk_score"),
        "blackout_probability": p,
        "markov_state": result.get("markov_state"),
    })

    skill_agents = {
        "anomaly_detector_skill": ("agents.anomaly_detector", "AnomalyDetectorSkill"),
        "renewables_forecast_skill": ("agents.renewables_forecast", "RenewablesForecastSkill"),
        "regulatory_compliance_skill": ("agents.regulatory_compliance", "RegulatoryComplianceSkill"),
    }
    result_key_by_skill = {
        "anomaly_detector_skill": "anomaly_report",
        "renewables_forecast_skill": "renewables_forecast",
        "regulatory_compliance_skill": "compliance_report",
    }

    for skill_name in plan["mandatory_skills"]:
        if skill_name in tool_names:
            continue
        module_name, attr_name = skill_agents[skill_name]
        import importlib
        skill_tool = getattr(importlib.import_module(module_name), attr_name)
        message = f"[ENFORCER] Analiza este contexto y genera tu output: {context_json}"
        output = _force(skill_tool.agent, skill_name, message, result, trajectory)
        result[result_key_by_skill[skill_name]] = output
        tool_names.add(skill_name)

    if plan["cross_border_mode"] != "none" and "cross_border_agent" not in tool_names:
        from agents.cross_border import cross_border_agent
        message = (
            f"[ENFORCER] mode='{plan['cross_border_mode']}'. "
            f"Evalua la capacidad de importacion. Contexto: {context_json}"
        )
        output = _force(cross_border_agent, "cross_border_agent", message, result, trajectory)
        result["cross_border"] = output
        tool_names.add("cross_border_agent")

    if plan["invoke_report_agent"] and "report_agent" not in tool_names:
        from agents.report_agent import report_agent
        message = (
            f"[ENFORCER] report_type='emergency'. Redacta el informe de emergencia. "
            f"Contexto: {context_json}"
        )
        _force(report_agent, "report_agent", message, result, trajectory)
        tool_names.add("report_agent")

    if plan["emit_alert"] != "none" and result.get("alert") is None:
        anomaly_report = result.get("anomaly_report") or {}
        risk = {
            "level": result.get("severity", "LOW"),
            "value": result.get("risk_score", 0),
            "blackout_probability": p,
        }
        result["alert"] = alert_formatter(
            risk=risk,
            anomalies=anomaly_report.get("anomalies", []),
            scope=plan["scope"],
        )

    result["trajectory"] = trajectory
    return result
