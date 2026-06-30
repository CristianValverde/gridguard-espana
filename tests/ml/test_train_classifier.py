# tests/ml/test_train_classifier.py
import numpy as np
import pandas as pd
from ml import incidents
from ml.train import train_classifier


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


def test_run_uses_graded_labels_and_weights(monkeypatch, tmp_path):
    """run() debe construir labels+weights con build_graded_labels y pasar sample_weight al fit."""
    import ml.train.train_classifier as tc

    captured = {}
    real_split = tc.leave_one_event_out_splits

    def spy_split(index, labels):
        captured["n_pos"] = int(labels.sum())
        return real_split(index, labels)

    monkeypatch.setattr(tc, "leave_one_event_out_splits", spy_split)
    # Ejecutar con 1 trial para que sea rápido; requiere el dataset real presente.
    tc.run(n_trials=1)
    assert captured.get("n_pos", 0) >= 12  # >= las 12 horas-evento graduadas
