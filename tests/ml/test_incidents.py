# tests/ml/test_incidents.py
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
