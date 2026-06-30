"""
ml/train/calibration.py — Calibración isotónica del clasificador de apagones.

Módulo separado para que _IsotonicCalibrated sea serializable por joblib/pickle
desde cualquier consumidor (validate_models, ml_models_server, retrain_pipeline).
Al vivir aquí, pickle registra la referencia como `ml.train.calibration._IsotonicCalibrated`
en lugar de `__main__._IsotonicCalibrated`, que falla al cargar fuera del proceso de entrenamiento.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

log = logging.getLogger("train_classifier")


class _IsotonicCalibrated:
    """Thin wrapper: XGBoost base + isotonic regression sobre P(blackout)."""

    def __init__(self, base_model, ir):
        self._base = base_model
        self._ir = ir

    def predict_proba(self, X):
        # Asegurar que la inferencia ocurre en CPU (el modelo pudo haberse entrenado en CUDA)
        self._base.set_params(device="cpu")
        raw = self._base.predict_proba(X)[:, 1]
        cal = self._ir.predict(raw)
        return np.column_stack([1.0 - cal, cal])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def feature_importances_(self):
        return self._base.feature_importances_


def calibrate_model(base_model, X_cal: np.ndarray, y_cal: np.ndarray):
    """Calibración isotónica sin CalibratedClassifierCV (incompatible con cv='prefit' en sklearn≥1.8)."""
    raw_proba = base_model.predict_proba(X_cal)[:, 1]
    brier_before = brier_score_loss(y_cal, raw_proba)
    log.info("Brier score antes de calibración: %.4f", brier_before)

    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(raw_proba, y_cal)

    cal_proba = ir.predict(raw_proba)
    brier_after = brier_score_loss(y_cal, cal_proba)
    log.info("Brier score después de calibración (isotónica): %.4f", brier_after)
    return _IsotonicCalibrated(base_model, ir)
