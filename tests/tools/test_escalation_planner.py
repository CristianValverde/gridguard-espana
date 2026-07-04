"""Tabla de verdad del escalation_planner: shift-left, sin LLM ni ADK.

Spec: docs/superpowers/specs/2026-07-04-escalation-planner-design.md
"""
from __future__ import annotations

import pytest

from tools.escalation_planner import escalation_planner


def test_verde_sin_markov_no_escala():
    plan = escalation_planner(blackout_probability=0.05, markov_state=0)
    assert plan["band"] == "VERDE"
    assert plan["mandatory_skills"] == []
    assert plan["cross_border_mode"] == "none"
    assert plan["invoke_report_agent"] is False
    assert plan["emit_alert"] == "none"


def test_verde_con_markov_s2_eleva_a_amarillo():
    plan = escalation_planner(blackout_probability=0.05, markov_state=2)
    assert plan["band"] == "AMARILLO"
    assert plan["mandatory_skills"] == ["anomaly_detector_skill"]
    assert "Markov" in plan["reason"]


def test_verde_con_markov_s4_tambien_eleva():
    plan = escalation_planner(blackout_probability=0.02, markov_state=4)
    assert plan["band"] == "AMARILLO"


def test_amarillo_sin_override_no_activa_cross_border():
    plan = escalation_planner(blackout_probability=0.20, markov_state=0)
    assert plan["band"] == "AMARILLO"
    assert plan["mandatory_skills"] == ["anomaly_detector_skill"]
    assert plan["cross_border_mode"] == "none"


def test_amarillo_con_markov_s1_activa_cross_border_assessment():
    plan = escalation_planner(blackout_probability=0.20, markov_state=1)
    assert plan["band"] == "AMARILLO"
    assert plan["cross_border_mode"] == "assessment"


def test_naranja_incluye_las_tres_skills_y_cross_border_assessment():
    plan = escalation_planner(blackout_probability=0.45, markov_state=0)
    assert plan["band"] == "NARANJA"
    assert plan["mandatory_skills"] == [
        "anomaly_detector_skill",
        "renewables_forecast_skill",
        "regulatory_compliance_skill",
    ]
    assert plan["cross_border_mode"] == "assessment"
    assert plan["invoke_report_agent"] is False
    assert plan["emit_alert"] == "warning"
    assert plan["scope"] == "peninsular"


def test_rojo_incluye_todo_y_report_agent():
    plan = escalation_planner(blackout_probability=0.75, markov_state=3)
    assert plan["band"] == "ROJO"
    assert plan["mandatory_skills"] == [
        "anomaly_detector_skill",
        "renewables_forecast_skill",
        "regulatory_compliance_skill",
    ]
    assert plan["cross_border_mode"] == "emergency"
    assert plan["invoke_report_agent"] is True
    assert plan["emit_alert"] == "critical"
    assert plan["scope"] == "nacional"


@pytest.mark.parametrize("p,expected_band", [
    (0.0, "VERDE"),
    (0.099999, "VERDE"),
    (0.10, "AMARILLO"),
    (0.299999, "AMARILLO"),
    (0.30, "NARANJA"),
    (0.599999, "NARANJA"),
    (0.60, "ROJO"),
    (1.0, "ROJO"),
])
def test_bordes_exactos_de_banda(p, expected_band):
    plan = escalation_planner(blackout_probability=p, markov_state=0)
    assert plan["band"] == expected_band


def test_probability_fuera_de_rango_se_recorta():
    plan = escalation_planner(blackout_probability=1.5, markov_state=0)
    assert plan["band"] == "ROJO"
    plan_neg = escalation_planner(blackout_probability=-0.5, markov_state=0)
    assert plan_neg["band"] == "VERDE"


def test_markov_state_fuera_de_rango_se_recorta():
    plan = escalation_planner(blackout_probability=0.05, markov_state=99)
    assert plan["band"] == "AMARILLO"  # recortado a 4 (>=2) con P<0.10
    plan_neg = escalation_planner(blackout_probability=0.05, markov_state=-3)
    assert plan_neg["band"] == "VERDE"  # recortado a 0
