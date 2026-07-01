import pandas as pd
import pytest
from ml.data import incremental_load as IL


def test_missing_hours_bootstrap_when_empty():
    now = pd.Timestamp("2026-07-01T12:30:00Z")
    hours = IL.missing_hours(None, now, bootstrap_days=2)
    assert hours[0] == pd.Timestamp("2026-06-29T12:00:00Z")
    assert hours[-1] == pd.Timestamp("2026-07-01T12:00:00Z")
    # continuo y horario
    assert (pd.Series(hours).diff().dropna() == pd.Timedelta(hours=1)).all()


def test_missing_hours_fills_gap_after_last():
    last = "2026-07-01T09:00:00Z"
    now = pd.Timestamp("2026-07-01T12:00:00Z")
    hours = IL.missing_hours(last, now)
    assert hours == [
        pd.Timestamp("2026-07-01T10:00:00Z"),
        pd.Timestamp("2026-07-01T11:00:00Z"),
        pd.Timestamp("2026-07-01T12:00:00Z"),
    ]


def test_missing_hours_empty_when_up_to_date():
    last = "2026-07-01T12:00:00Z"
    now = pd.Timestamp("2026-07-01T12:30:00Z")
    assert IL.missing_hours(last, now) == []


def test_raw_to_snapshots_maps_training_scale_columns():
    idx = pd.date_range("2025-06-01", periods=2, freq="h", tz="UTC")
    df = pd.DataFrame(index=idx)
    df["demand_real_mw"] = [26000.0, 25500.0]
    df["wind_mw"] = [5000.0, 5200.0]
    df["solar_pv_mw"] = [8000.0, 7800.0]
    df["nuclear_mw"] = [7000.0, 7000.0]
    df["es_fr_mw"] = [-600.0, -700.0]
    df["es_pt_mw"] = [-300.0, -250.0]
    df["es_ma_mw"] = [50.0, 40.0]
    df["secondary_reserve_up_mwh"] = [120.0, 90.0]
    df["secondary_reserve_dn_mwh"] = [-80.0, -60.0]

    snaps = IL.raw_to_snapshots(df)
    assert len(snaps) == 2
    s0 = snaps[0]
    assert s0["demand_mw"] == 26000.0
    assert s0["generation_by_tech"]["wind_mw"] == 5000.0
    assert s0["interchange_flows"]["es_fr_mw"] == -600.0
    # reserve_mw = |up| + |dn|  (misma fórmula que build_dataset)
    assert s0["reserve_mw"] == 200.0
    # timestamp ISO UTC, parseable por serve_features
    assert pd.Timestamp(s0["timestamp"]) == idx[0]


def test_complete_snapshots_drops_zero_generation():
    snaps = [
        {"generation_by_tech": {"nuclear_mw": 7000.0, "wind_mw": 500.0}},   # completo
        {"generation_by_tech": {"nuclear_mw": 0.0, "wind_mw": 0.0}},        # incompleto (no publicado)
        {"generation_by_tech": {}},                                          # vacío
    ]
    kept = IL.complete_snapshots(snaps)
    assert len(kept) == 1
    assert kept[0]["generation_by_tech"]["nuclear_mw"] == 7000.0


def test_raw_to_snapshots_handles_nan():
    idx = pd.date_range("2025-06-01", periods=1, freq="h", tz="UTC")
    df = pd.DataFrame(index=idx)
    df["demand_real_mw"] = [float("nan")]
    df["wind_mw"] = [float("nan")]
    snaps = IL.raw_to_snapshots(df)
    assert snaps[0]["demand_mw"] == 0.0
    assert snaps[0]["generation_by_tech"]["wind_mw"] == 0.0
