# tests/agents/test_anomaly_detector_ext.py
from schemas.models import AnomalyReport


def test_anomaly_report_has_incident_type_and_causal_chain():
    r = AnomalyReport(
        anomalies=[], severity="MEDIUM", confidence=0.5, ml_score=0.4,
        regulatory_context="", reasoning="x",
        incident_type="renewable_ramp", causal_chain="subida solar brusca → balance positivo",
    )
    assert r.incident_type == "renewable_ramp"


def test_instruction_mentions_incident_classification():
    import agents.anomaly_detector as ad
    assert "incident_type" in ad._INSTRUCTION
    assert "get_grid_state" in ad._INSTRUCTION


def test_module_imports():
    import agents.anomaly_detector as ad
    assert ad.AnomalyDetectorSkill is not None
