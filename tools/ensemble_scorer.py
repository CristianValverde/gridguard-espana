"""tools/ensemble_scorer.py — Fusión ensemble del riesgo principal.

Combina tres detectores con sesgos inductivos distintos:
  - LSTM autoencoder (no supervisado, inmune a D5) → load-bearing.
  - Isolation Forest (no supervisado, anomalía puntual) → corroboración.
  - Clasificador supervisado (capado por D5: 3 eventos) → voto secundario.

Los pesos están dominados por el LSTM POR DISEÑO (no ajustados al golden), para
mitigar el overfitting con n=1 evento S4. El objetivo es que el modelo que sí
funciona (LSTM, que discrimina el golden) conduzca el riesgo, con los otros dos
como corroboración — en vez de dejar el riesgo en manos del clasificador de 3
eventos.
"""
from __future__ import annotations

DEFAULT_WEIGHTS = {"lstm": 0.5, "if": 0.2, "clf": 0.3}
OVERRIDE_THRESHOLD = 0.9   # lstm_recon_norm ∈ [0,1]; ≥ esto = anomalía temporal fuerte
OVERRIDE_FLOOR = 0.6       # suelo de P(blackout) cuando el override dispara


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def fuse(lstm_recon_norm: float, if_score: float, clf_proba: float,
         weights: dict | None = None) -> dict:
    """Fusiona las tres señales en una P(blackout) ∈ [0, 1].

    Args:
        lstm_recon_norm: error de reconstrucción del LSTM normalizado a [0,1]
            (típicamente min(mse/threshold, 1)).
        if_score:        anomaly score del Isolation Forest en [0,1].
        clf_proba:       P(blackout) del clasificador supervisado en [0,1].
        weights:         pesos opcionales (default DEFAULT_WEIGHTS, LSTM dominante).

    Returns:
        dict con blackout_probability, primary='lstm', override_active y components.
    """
    w = weights or DEFAULT_WEIGHTS
    lstm = _clip01(lstm_recon_norm)
    isof = _clip01(if_score)
    clf = _clip01(clf_proba)

    p = w["lstm"] * lstm + w["if"] * isof + w["clf"] * clf
    p = _clip01(p)

    override = lstm >= OVERRIDE_THRESHOLD
    if override:
        p = max(p, OVERRIDE_FLOOR)

    return {
        "blackout_probability": round(p, 4),
        "primary": "lstm",
        "override_active": override,
        "components": {"lstm": round(lstm, 4), "if": round(isof, 4), "clf": round(clf, 4)},
    }
