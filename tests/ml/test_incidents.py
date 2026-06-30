# tests/ml/test_incidents.py
import numpy as np
import pandas as pd
from ml import incidents


def test_apagon_timestamp_is_corrected_to_utc_1033():
    """El colapso del 28-abr-2025 fue a las 10:33 UTC (12:33 CEST), no a las 08:00."""
    by_label = {i["label"]: i for i in incidents.documented_incidents_utc()}
    apagon = next(i for i in incidents.documented_incidents_utc()
                  if i["severity"] == incidents.S4_BLACKOUT)
    assert apagon["ts"] == pd.Timestamp("2025-04-28 10:33", tz="UTC")
    assert apagon["ts"].tz is not None  # tz-aware


def test_three_documented_events_in_training_range():
    """Deben existir los 3 eventos mayores 2021–2025, todos dentro de 2020–2026 UTC."""
    incs = incidents.documented_incidents_utc()
    assert len(incs) == 3
    lo = pd.Timestamp("2020-01-01", tz="UTC")
    hi = pd.Timestamp("2026-06-30", tz="UTC")
    assert all(lo <= i["ts"] <= hi for i in incs)
    severities = sorted(i["severity"] for i in incs)
    assert severities == [incidents.S3_MAJOR, incidents.S3_MAJOR, incidents.S4_BLACKOUT]


def test_graded_labels_mark_window_before_each_event():
    # Índice horario que cubre los 3 eventos
    index = pd.date_range("2021-01-01", "2025-05-01", freq="h", tz="UTC")
    labels, weights = incidents.build_graded_labels(index)

    # La ventana del apagón es [04:33, 10:33) UTC del 28-abr-2025 → 6 horas positivas
    apagon_window = (index >= pd.Timestamp("2025-04-28 04:33", tz="UTC")) & \
                    (index < pd.Timestamp("2025-04-28 10:33", tz="UTC"))
    assert labels[apagon_window].sum() == apagon_window.sum() > 0

    # NINGÚN positivo en la madrugada tranquila (02:00–03:00 UTC) — el bug viejo
    quiet = (index >= pd.Timestamp("2025-04-28 02:00", tz="UTC")) & \
            (index < pd.Timestamp("2025-04-28 03:00", tz="UTC"))
    assert labels[quiet].sum() == 0

    # Peso del apagón (S4) = 3.0; peso de un evento S3 = 2.0
    assert weights[apagon_window].max() == 3.0
    s3_window = (index >= pd.Timestamp("2021-07-24 11:35", tz="UTC")) & \
                (index < pd.Timestamp("2021-07-24 14:35", tz="UTC"))
    assert weights[s3_window].max() == 2.0


def test_graded_labels_have_more_positives_than_old_single_window():
    index = pd.date_range("2021-01-01", "2025-05-01", freq="h", tz="UTC")
    labels, _ = incidents.build_graded_labels(index)
    # 6h (S4) + 3h + 3h (2×S3) = 12 horas-evento positivas
    assert labels.sum() == 12
