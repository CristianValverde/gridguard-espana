import pytest
from tools.ensemble_scorer import (
    DEFAULT_WEIGHTS, OVERRIDE_FLOOR, OVERRIDE_THRESHOLD, fuse, normalize_if_signal,
)


def test_output_bounds_and_keys():
    out = fuse(0.2, 0.3, 0.1)
    assert 0.0 <= out["blackout_probability"] <= 1.0
    assert out["primary"] == "lstm"
    assert {"lstm", "if", "clf"} <= set(out["components"].keys())


def test_lstm_is_dominant():
    """Subir solo el LSTM mueve el score más que subir solo el CLF la misma cantidad."""
    base = fuse(0.0, 0.0, 0.0)["blackout_probability"]
    d_lstm = fuse(0.5, 0.0, 0.0)["blackout_probability"] - base
    d_clf = fuse(0.0, 0.0, 0.5)["blackout_probability"] - base
    assert d_lstm > d_clf


def test_lstm_override_floors_probability():
    """LSTM muy por encima de su umbral eleva el suelo aunque IF/CLF sean bajos."""
    out = fuse(1.0, 0.0, 0.0)
    assert out["blackout_probability"] >= OVERRIDE_FLOOR
    assert out["override_active"] is True


def test_no_override_below_threshold():
    out = fuse(OVERRIDE_THRESHOLD - 0.01, 0.0, 0.0)
    assert out["override_active"] is False


def test_inputs_clipped_to_unit_interval():
    out = fuse(5.0, -3.0, 2.0)  # fuera de [0,1]
    assert 0.0 <= out["blackout_probability"] <= 1.0
    assert out["components"]["lstm"] == 1.0
    assert out["components"]["if"] == 0.0
    assert out["components"]["clf"] == 1.0


def test_custom_weights_respected():
    w = {"lstm": 0.0, "if": 1.0, "clf": 0.0}
    out = fuse(1.0, 0.5, 1.0, weights=w)
    # solo el IF pesa (0.5); sin override porque lstm=1.0 dispararía... comprobamos componente
    assert out["components"]["if"] == 0.5


def test_normalize_if_signal_centered_on_model_threshold():
    """df=0 (umbral de contaminación) → 0.5; anomalía clara → 1; inlier claro → 0.
    Garantiza que la señal IF no satura constante (bug de la normalización legacy)."""
    assert normalize_if_signal(0.0) == pytest.approx(0.5)
    assert normalize_if_signal(-0.1) == pytest.approx(1.0)
    assert normalize_if_signal(0.1) == pytest.approx(0.0)
    assert normalize_if_signal(-5.0) == 1.0 and normalize_if_signal(5.0) == 0.0
    # monótona decreciente en df
    assert normalize_if_signal(-0.05) > normalize_if_signal(0.05)
