"""Paridad train/serve en ml_models_server (Task 10).

Con bundles nuevos (scaler / feature_cols), get_anomaly_score y
get_blackout_probability deben construir sus vectores vía ml.serve_features,
no con la matriz raw legacy. Tests offline con modelos stub.
"""
import json

import numpy as np
import pandas as pd
import pytest

from mcp_servers import ml_models_server as S
from ml.features import FEATURE_COLS


def _snapshots(n=48):
    idx = pd.date_range("2025-06-01", periods=n, freq="h", tz="UTC")
    snaps = []
    for i, ts in enumerate(idx):
        r = i / n
        snaps.append({
            "timestamp": ts.isoformat(),
            "demand_mw": 30000.0,
            "generation_by_tech": {
                "nuclear_mw": 7000.0, "hydro_mw": 2000.0, "gas_cc_mw": 3000.0,
                "coal_mw": 0.0, "cogeneration_mw": 1500.0, "pumping_mw": 0.0,
                "wind_mw": 8000.0 + 4000.0 * r, "solar_pv_mw": 6000.0 + 6000.0 * r,
                "solar_thermal_mw": 500.0,
            },
            "interchange_flows": {"es_fr_mw": -1400.0 * r, "es_pt_mw": -300.0, "es_ma_mw": 50.0},
            "reserve_mw": 400.0,
        })
    return snaps


class _StubIF:
    """Isolation Forest de mentira que registra la matriz que recibe."""

    def __init__(self):
        self.seen = None

    def decision_function(self, X):
        self.seen = np.asarray(X)
        return np.full(len(self.seen), 0.05)  # inlier suave

    def predict(self, X):
        return np.ones(len(np.asarray(X)), dtype=int)


class _StubCLF:
    def __init__(self):
        self.seen = None

    def predict_proba(self, X):
        self.seen = np.asarray(X)
        return np.array([[0.3, 0.7]])


def test_anomaly_score_bundle_scaler_uses_serve_features(monkeypatch):
    stub = _StubIF()
    scaler_feats = ["balance_mw", "renewables_pct", "non_synchronous_penetration"]
    bundle = {"model": stub, "scaler": {
        "features": scaler_feats,
        "mean": {f: 0.0 for f in scaler_feats},
        "std": {f: 1.0 for f in scaler_feats},
    }}
    monkeypatch.setitem(S._models, "iforest", bundle)
    monkeypatch.setitem(S._models, "lstm_ae", None)

    out = S.get_anomaly_score(json.dumps(_snapshots()))
    assert "error" not in out, out
    assert 0.0 <= out["if_score"] <= 1.0
    # el vector tiene las features del scaler (derivadas), no las 9 raw legacy
    assert stub.seen.shape[1] == len(scaler_feats)
    assert set(out["top_features"]) <= set(FEATURE_COLS)


def test_anomaly_score_legacy_artifact_keeps_raw_matrix(monkeypatch):
    stub = _StubIF()
    monkeypatch.setitem(S._models, "iforest", stub)  # artefacto viejo: modelo pelado
    monkeypatch.setitem(S._models, "lstm_ae", None)

    out = S.get_anomaly_score(json.dumps(_snapshots()))
    assert "error" not in out, out
    assert stub.seen.shape[1] == 9  # matriz raw legacy


def test_blackout_probability_bundle_uses_feature_cols(monkeypatch):
    stub = _StubCLF()
    bundle = {"model": stub, "feature_cols": list(FEATURE_COLS)}
    monkeypatch.setitem(S._models, "xgb_blackout", bundle)

    row = {c: 1.0 for c in FEATURE_COLS}
    out = S.get_blackout_probability(json.dumps(row))
    assert "error" not in out, out
    assert abs(out["probability"] - 0.7) < 1e-9
    assert stub.seen.shape == (1, len(FEATURE_COLS))


def test_anomaly_result_schema_accepts_combined_model_type():
    """Con IF y LSTM disponibles a la vez, model_type es 'isolation_forest+lstm_ae'."""
    from schemas.models import AnomalyResult

    r = AnomalyResult(anomaly_score=0.5, is_anomaly=False, top_features=[],
                      feature_scores={}, model_type="isolation_forest+lstm_ae",
                      inference_time_ms=0.0)
    assert r.model_type == "isolation_forest+lstm_ae"


def test_isotonic_calibrated_feature_importances_is_iterable():
    """feature_importances_ debe ser property (no método) para que
    zip(names, model.feature_importances_) funcione en ml_models_server."""
    import numpy as np
    from ml.train.calibration import _IsotonicCalibrated

    class _Base:
        feature_importances_ = np.array([0.5, 0.5])

        def set_params(self, **kw):
            return self

    wrapped = _IsotonicCalibrated(_Base(), ir=None)
    assert list(wrapped.feature_importances_) == [0.5, 0.5]


def test_ensemble_reads_probability_key_from_blackout_risk(monkeypatch):
    """BlackoutRisk expone 'probability' — el ensemble no debe leer 0."""
    monkeypatch.setattr(S, "get_anomaly_score",
                        lambda _w: {"if_score": 0.0, "lstm_recon_norm": 0.0})
    monkeypatch.setattr(S, "get_blackout_probability",
                        lambda _f: {"probability": 0.9, "risk_level": "CRITICAL"})
    out = S.get_ensemble_score("{}", "[]")
    assert out["components"]["clf"] == 0.9
