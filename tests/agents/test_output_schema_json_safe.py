"""Los output_schema de los agentes deben sobrevivir a json.dumps.

SetModelResponseTool de ADK 2.3.0 serializa el resultado con
model_dump() en modo python + json.dumps sin default: un campo datetime
rompe el nodo con TypeError (visto en la corrida live 2026-07-03 en
cross_border_agent). Estos tests reproducen esa ruta de serializacion.
"""
from __future__ import annotations

import json
from datetime import datetime

from schemas.models import (
    AnomalyReport,
    ComplianceReport,
    CrossBorderAnalysis,
    Lesson,
    RenewablesForecast,
)


def _instances():
    return [
        AnomalyReport(
            anomalies=[], severity="LOW", confidence=0.5, ml_score=0.1,
            regulatory_context="", reasoning="x",
        ),
        RenewablesForecast(
            risk_level="LOW", expected_generation_pct=40.0,
            critical_window="14:00-18:00 UTC", ml_forecast_mw=25000.0,
            reasoning="x",
        ),
        ComplianceReport(compliance_status="compliant", reasoning="x"),
        CrossBorderAnalysis(
            can_import=True, available_capacity_mw=1000.0,
            countries=["FR"], risk_mitigation="x",
        ),
        Lesson(
            lesson="Escalar a AMARILLO aunque P<0.10 si Markov ya está en S2.",
            context="Ciclo 2026-07-04T10:00Z: P=0.08 pero Markov=S2 tras rampa solar.",
            markov_state=2, p_band="AMARILLO",
        ),
    ]


def test_output_schemas_json_serializables_como_hace_set_model_response():
    for model in _instances():
        dumped = model.model_dump(exclude_none=True)  # mismo modo que ADK
        json.dumps(dumped, ensure_ascii=False)  # no debe lanzar TypeError


def test_timestamp_default_es_str_iso8601_parseable():
    for model in _instances():
        assert isinstance(model.timestamp, str), type(model).__name__
        datetime.fromisoformat(model.timestamp)  # parseable, no lanza
