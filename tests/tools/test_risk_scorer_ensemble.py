from tools.risk_scorer import risk_scorer

_SNAP = {"demand_mw": 30000, "generation_by_tech": {"wind_mw": 14000, "solar_pv_mw": 13000},
         "interchange_flows": {"es_fr_mw": -1800}, "reserve_mw": 300}


def test_fused_score_drives_blackout_probability():
    """El score fusionado (ml_score) se propaga a blackout_probability y sube el nivel."""
    out = risk_scorer(_SNAP, ml_score=0.82)
    assert abs(out["blackout_probability"] - 0.82) < 1e-3
    assert out["level"] in ("HIGH", "CRITICAL")


def test_low_fused_score_stays_low():
    out = risk_scorer(_SNAP, ml_score=0.02)
    assert out["blackout_probability"] < 0.1
    assert out["level"] in ("LOW", "MEDIUM")
