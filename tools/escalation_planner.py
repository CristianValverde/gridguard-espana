"""tools/escalation_planner.py — Function Tool: política de escalación
determinista (shift-left: Python puro, sin LLM).

Fuente de verdad de PASO 2 del prompt del orchestrator (prompts/orchestrator.md).
Reemplaza la prosa: la política vive aquí, testeable offline.

Spec: docs/superpowers/specs/2026-07-04-escalation-planner-design.md
"""

from __future__ import annotations

_ANOMALY = "anomaly_detector_skill"
_RENEWABLES = "renewables_forecast_skill"
_COMPLIANCE = "regulatory_compliance_skill"


def escalation_planner(blackout_probability: float, markov_state: int) -> dict:
    """Calcula el plan de escalación para un ciclo de monitorización.

    Args:
        blackout_probability: P(blackout) del ensemble ML, se recorta a [0.0, 1.0].
        markov_state: Estado Markov actual S0-S4 (0-4), se recorta a ese rango.

    Returns:
        dict con band, mandatory_skills, cross_border_mode, invoke_report_agent,
        emit_alert, scope, reason.
    """
    p = min(max(float(blackout_probability), 0.0), 1.0)
    state = min(max(int(markov_state), 0), 4)
    reason_parts: list[str] = []

    if p < 0.10:
        band = "VERDE"
    elif p < 0.30:
        band = "AMARILLO"
    elif p < 0.60:
        band = "NARANJA"
    else:
        band = "ROJO"

    # Override 1: Markov S2+ eleva VERDE a AMARILLO aunque P sea baja.
    if band == "VERDE" and state >= 2:
        band = "AMARILLO"
        reason_parts.append(f"Markov S{state} eleva VERDE→AMARILLO pese a P={p:.2f}")

    cross_border_mode = "none"
    if band == "AMARILLO":
        mandatory_skills = [_ANOMALY]
        # Override 2: Markov S1 en banda AMARILLO activa assessment proactivo.
        if state == 1 and 0.10 <= p < 0.30:
            cross_border_mode = "assessment"
            reason_parts.append("Markov S1 en AMARILLO activa cross_border assessment")
    elif band == "NARANJA":
        mandatory_skills = [_ANOMALY, _RENEWABLES, _COMPLIANCE]
        cross_border_mode = "assessment"
    elif band == "ROJO":
        mandatory_skills = [_ANOMALY, _RENEWABLES, _COMPLIANCE]
        cross_border_mode = "emergency"
    else:  # VERDE sin override
        mandatory_skills = []

    invoke_report_agent = band == "ROJO"

    if band == "ROJO":
        emit_alert = "critical"
        scope = "nacional"
    elif band == "NARANJA":
        emit_alert = "warning"
        scope = "peninsular"
    else:
        emit_alert = "none"
        scope = "peninsular"

    if not reason_parts:
        reason_parts.append(f"Banda {band} por P={p:.2f}, Markov S{state}")

    return {
        "band": band,
        "mandatory_skills": mandatory_skills,
        "cross_border_mode": cross_border_mode,
        "invoke_report_agent": invoke_report_agent,
        "emit_alert": emit_alert,
        "scope": scope,
        "reason": "; ".join(reason_parts),
    }
