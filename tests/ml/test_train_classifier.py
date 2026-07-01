# tests/ml/test_train_classifier.py
import joblib
import numpy as np
import pandas as pd
import pytest
from ml import incidents
from ml.train import train_classifier
from ml.train.calibration import _IsotonicCalibrated

# El dataset de entrenamiento es un artefacto regenerable (gitignored). Los tests
# que llaman a run() lo requieren: se auto-omiten si no está presente (checkout
# limpio) y corren donde exista (regenerar: python -m ml.data.build_dataset).
_requires_dataset = pytest.mark.skipif(
    not train_classifier.DATA_PATH.exists(),
    reason=f"dataset no presente: {train_classifier.DATA_PATH} "
           "(regenerable vía `python -m ml.data.build_dataset`)",
)


def test_loeo_every_val_fold_has_a_positive():
    index = pd.date_range("2021-01-01", "2025-05-01", freq="h", tz="UTC")
    labels, _ = incidents.build_graded_labels(index)
    y = labels.values
    splits = train_classifier.leave_one_event_out_splits(index, y)
    assert len(splits) == 3  # un fold por evento documentado
    for train_mask, val_mask in splits:
        assert y[val_mask].sum() > 0, "fold de validación sin positivos"
        assert y[train_mask].sum() > 0, "fold de entrenamiento sin positivos"
        # train y val son disjuntos
        assert not (train_mask & val_mask).any()


@_requires_dataset
def test_run_uses_graded_labels_and_weights(monkeypatch, tmp_path):
    """run() debe construir labels+weights con build_graded_labels y pasar sample_weight al fit."""
    import ml.train.train_classifier as tc

    # Redirigir la salida a tmp_path para no sobreescribir el artefacto real
    monkeypatch.setattr(tc, "CANDIDATES_DIR", tmp_path)

    captured = {}
    real_split = tc.leave_one_event_out_splits

    def spy_split(index, labels):
        captured["n_pos"] = int(labels.sum())
        return real_split(index, labels)

    monkeypatch.setattr(tc, "leave_one_event_out_splits", spy_split)
    # Ejecutar con 1 trial para que sea rápido; requiere el dataset real presente.
    tc.run(n_trials=1)
    assert captured.get("n_pos", 0) >= 12  # >= las 12 horas-evento graduadas
    # Confirmar que el modelo se guardó en tmp_path y no en el directorio real
    assert (tmp_path / "blackout_classifier.pkl").exists()


@_requires_dataset
def test_bundle_serialization_module(monkeypatch, tmp_path):
    """El bundle debe desserializarse con __class__.__module__ == 'ml.train.calibration'.

    Reproduce el bug original: cuando _IsotonicCalibrated se definía en __main__,
    pickle registraba 'ml.train.calibration._IsotonicCalibrated' debería ser correcto
    pero cualquier consumer que no sea el proceso de entrenamiento fallaba con
    AttributeError: module '__main__' has no attribute '_IsotonicCalibrated'.
    """
    import ml.train.train_classifier as tc

    monkeypatch.setattr(tc, "CANDIDATES_DIR", tmp_path)
    tc.run(n_trials=1)

    pkl_path = tmp_path / "blackout_classifier.pkl"
    assert pkl_path.exists(), "El bundle no se guardó en tmp_path"

    # Cargar desde disco — simula exactamente lo que hacen validate_models, ml_models_server, etc.
    bundle = joblib.load(pkl_path)
    model = bundle["model"]

    # La aserción clave: el módulo debe ser el módulo importable, NO '__main__'
    assert model.__class__.__module__ == "ml.train.calibration", (
        f"Módulo incorrecto: {model.__class__.__module__!r}. "
        "Indica que _IsotonicCalibrated fue serializado desde __main__ "
        "y no puede ser deserializado por consumidores externos."
    )
    assert isinstance(model, _IsotonicCalibrated)

    # Verificar que predict_proba funciona sobre un array dummy del ancho correcto
    # Usar feature_cols del bundle (incluye lag features, más ancho que FEATURE_COLS base)
    n_features = len(bundle["feature_cols"])
    X_dummy = np.zeros((3, n_features), dtype=np.float32)
    proba = model.predict_proba(X_dummy)
    assert proba.shape == (3, 2), f"predict_proba shape inesperado: {proba.shape}"
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6), "Las probabilidades no suman 1"
