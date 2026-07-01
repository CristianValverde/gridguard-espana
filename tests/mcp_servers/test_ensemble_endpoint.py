import pytest
from mcp_servers import ml_models_server as S


def test_ensemble_degrades_without_models(monkeypatch):
    """Sin modelos (señales a 0), el endpoint no crashea y devuelve P en [0,1]."""
    monkeypatch.setattr(S, "get_anomaly_score", lambda _w: {"if_score": 0.0, "lstm_recon_norm": 0.0})
    monkeypatch.setattr(S, "get_blackout_probability", lambda _f: {"blackout_probability": 0.0})
    out = S.get_ensemble_score("{}", "[]")
    assert 0.0 <= out["blackout_probability"] <= 1.0
    assert out["primary"] == "lstm"


def test_ensemble_fuses_three_signals(monkeypatch):
    monkeypatch.setattr(S, "get_anomaly_score", lambda _w: {"if_score": 0.4, "lstm_recon_norm": 0.8})
    monkeypatch.setattr(S, "get_blackout_probability", lambda _f: {"blackout_probability": 0.3})
    out = S.get_ensemble_score("{}", "[]")
    # 0.5*0.8 + 0.2*0.4 + 0.3*0.3 = 0.57
    assert abs(out["blackout_probability"] - 0.57) < 1e-6
    assert out["components"] == {"lstm": 0.8, "if": 0.4, "clf": 0.3}


def test_ensemble_tolerates_error_dicts(monkeypatch):
    """Si un modelo devuelve {'error': ...}, su señal cuenta como 0 (degradación)."""
    monkeypatch.setattr(S, "get_anomaly_score", lambda _w: {"error": "Modelos no entrenados"})
    monkeypatch.setattr(S, "get_blackout_probability", lambda _f: {"blackout_probability": 0.9})
    out = S.get_ensemble_score("{}", "[]")
    assert out["components"]["lstm"] == 0.0 and out["components"]["if"] == 0.0
    assert out["components"]["clf"] == 0.9
