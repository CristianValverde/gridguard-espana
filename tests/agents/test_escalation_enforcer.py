"""agents/escalation_enforcer.py: repara por codigo la infra-escalacion del
orchestrator cuando el LLM no ejecuta todo lo que escalation_planner manda.

Spec: docs/superpowers/specs/2026-07-04-escalation-enforcer-design.md
"""
from __future__ import annotations

from agents.escalation_enforcer import enforce_escalation_plan


def _base_result(**overrides) -> dict:
    base = {
        "timestamp": "2026-07-04T10:00:00+00:00",
        "risk_score": 85,
        "blackout_probability": 0.75,
        "markov_state": "S3",
        "severity": "CRITICAL",
        "scope": "nacional",
        "alert": None,
        "anomaly_report": None,
        "renewables_forecast": None,
        "compliance_report": None,
        "cross_border": None,
        "actions_taken": ["ree_poller", "risk_scorer"],
    }
    base.update(overrides)
    return base


def test_rojo_fuerza_las_skills_y_subagentes_faltantes(monkeypatch):
    calls = []

    def _fake_run_agent_once(agent, message_text):
        calls.append(getattr(agent, "name", str(agent)))
        name = getattr(agent, "name", "")
        if name == "anomaly_detector_skill":
            output = {"anomalies": [], "severity": "HIGH", "confidence": 0.8,
                       "ml_score": 0.7, "regulatory_context": "x", "reasoning": "x"}
        elif name == "renewables_forecast_skill":
            output = {"risk_level": "HIGH", "expected_generation_pct": 60.0,
                       "critical_window": "14:00-18:00 UTC", "ml_forecast_mw": 30000.0,
                       "reasoning": "x"}
        elif name == "regulatory_compliance_skill":
            output = {"compliance_status": "warning", "reasoning": "x"}
        elif name == "cross_border_agent":
            output = {"can_import": True, "available_capacity_mw": 500.0,
                       "countries": ["FR"], "risk_mitigation": "x"}
        else:  # report_agent
            output = {"raw": "informe redactado"}
        trajectory = [{"tool": f"{name}_internal_tool", "args": {}}] if name in (
            "cross_border_agent", "report_agent") else []
        return {"output": output, "trajectory": trajectory}

    monkeypatch.setattr(
        "agents.escalation_enforcer._run_agent_once", _fake_run_agent_once
    )

    result = _base_result()
    trajectory = [{"tool": "anomaly_detector_skill", "args": {}}]  # el LLM solo llego hasta aqui

    out = enforce_escalation_plan(result, trajectory, snapshot={"demand_mw": 32000})

    tools = {step["tool"] for step in out["trajectory"] if not step["tool"].endswith("_internal_tool")}
    assert "renewables_forecast_skill" in tools
    assert "regulatory_compliance_skill" in tools
    assert "cross_border_agent" in tools
    assert "report_agent" in tools
    # anomaly_detector_skill ya estaba, NO debe forzarse de nuevo
    assert "anomaly_detector_skill" not in calls

    assert out["renewables_forecast"]["risk_level"] == "HIGH"
    assert out["compliance_report"]["compliance_status"] == "warning"
    assert out["cross_border"]["can_import"] is True
    assert any("forzado por enforcer" in a for a in out["actions_taken"])

    # Bubbling: las tool-calls internas de los sub_agents forzados (no de las skills) aparecen
    internal_tools = {step["tool"] for step in out["trajectory"]}
    assert "cross_border_agent_internal_tool" in internal_tools
    assert "report_agent_internal_tool" in internal_tools


def test_verde_no_fuerza_nada():
    result = _base_result(
        risk_score=5, blackout_probability=0.03, markov_state="S0",
        severity="LOW", scope="peninsular",
    )
    trajectory = []

    out = enforce_escalation_plan(result, trajectory, snapshot={})

    assert out["trajectory"] == []
    assert out["anomaly_report"] is None
    assert not any("forzado por enforcer" in a for a in out["actions_taken"])


def test_naranja_con_trayectoria_completa_es_idempotente(monkeypatch):
    def _boom(agent, message_text):
        raise AssertionError("no deberia forzar nada: la trayectoria ya esta completa")

    monkeypatch.setattr("agents.escalation_enforcer._run_agent_once", _boom)

    result = _base_result(
        risk_score=45, blackout_probability=0.40, markov_state="S1",
        severity="MEDIUM", scope="peninsular",
        alert={"severity": "MEDIUM", "scope": "peninsular", "timestamp": "x",
               "summary": "x", "details": "x", "recommended_action": "x"},
    )
    trajectory = [
        {"tool": "anomaly_detector_skill", "args": {}},
        {"tool": "renewables_forecast_skill", "args": {}},
        {"tool": "regulatory_compliance_skill", "args": {}},
        {"tool": "cross_border_agent", "args": {}},
    ]

    out = enforce_escalation_plan(result, trajectory, snapshot={})

    assert out["trajectory"] == trajectory  # sin cambios


def test_alerta_se_rellena_con_alert_formatter_puro_sin_llm():
    result = _base_result(
        risk_score=85, blackout_probability=0.75, markov_state="S3",
        severity="CRITICAL", scope="nacional", alert=None,
        anomaly_report={"anomalies": [{"feature": "demand_mw", "value": 32000.0,
                                        "deviation_sigma": 3.2, "description": "pico"}],
                        "severity": "HIGH", "confidence": 0.8, "ml_score": 0.7,
                        "regulatory_context": "x", "reasoning": "x"},
    )
    trajectory = [
        {"tool": "anomaly_detector_skill", "args": {}},
        {"tool": "renewables_forecast_skill", "args": {}},
        {"tool": "regulatory_compliance_skill", "args": {}},
        {"tool": "cross_border_agent", "args": {}},
        {"tool": "report_agent", "args": {}},
    ]

    out = enforce_escalation_plan(result, trajectory, snapshot={})

    assert out["alert"] is not None
    assert out["alert"]["severity"] == "CRITICAL"
    assert out["alert"]["scope"] == "nacional"


def test_markov_state_mal_formado_no_lanza_excepcion():
    result = _base_result(markov_state="?", blackout_probability=0.5)
    out = enforce_escalation_plan(result, trajectory=[], snapshot={})
    assert isinstance(out, dict)  # no lanza
