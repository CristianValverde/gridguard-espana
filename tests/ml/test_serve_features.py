import numpy as np
import pandas as pd
import pytest
from ml import features as F
from ml import serve_features as SF


def _full_dataset(n=400):
    """DataFrame horario continuo con columnas raw, como el dataset de entrenamiento."""
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    r = np.linspace(0, 1, n)
    df = pd.DataFrame(index=idx)
    df.index.name = "timestamp"
    df["demand_real_mw"] = 30000.0 + 2000.0 * np.sin(np.linspace(0, 20, n))
    df["nuclear_mw"] = 7000.0
    df["hydro_mw"] = 2000.0
    df["gas_cc_mw"] = 3000.0 - 1000.0 * r
    df["coal_mw"] = 0.0
    df["cogeneration_mw"] = 1500.0
    df["pumping_mw"] = 0.0
    df["wind_mw"] = 8000.0 + 4000.0 * r
    df["solar_pv_mw"] = 6000.0 + 6000.0 * r
    df["solar_thermal_mw"] = 500.0
    df["es_fr_mw"] = -1400.0 * r
    df["es_pt_mw"] = -300.0 + 200.0 * np.cos(np.linspace(0, 10, n))
    df["es_ma_mw"] = 50.0
    df["reserve_mw"] = 400.0 + 100.0 * r
    return df


def _df_to_snapshots(df):
    snaps = []
    for ts, row in df.iterrows():
        snaps.append({
            "timestamp": ts.isoformat(),
            "demand_mw": float(row["demand_real_mw"]),
            "generation_by_tech": {c: float(row[c]) for c in F.GEN_COLS},
            "interchange_flows": {c: float(row[c]) for c in ("es_fr_mw", "es_pt_mw", "es_ma_mw")},
            "reserve_mw": float(row["reserve_mw"]),
        })
    return snaps


def test_serve_row_matches_train_path_at_same_timestamp():
    """PARIDAD: la fila de serve (ventana desde snapshots) == fila de train
    (build_features sobre el dataset completo) en el mismo timestamp `now`."""
    df_full = _full_dataset(400)
    now = df_full.index[300]  # con >168h de historia por detrás

    train_feats = F.build_features(df_full)
    train_row = train_feats.loc[now]

    snaps = _df_to_snapshots(df_full.loc[:now])  # historia hasta `now`
    serve_row = SF.serve_feature_row(snaps, now)

    for col in F.FEATURE_COLS:
        assert abs(float(serve_row[col]) - float(train_row[col])) < 1e-9, (
            f"desajuste train/serve en {col}: "
            f"serve={serve_row[col]} train={train_row[col]}"
        )


def test_serve_row_cold_start_does_not_crash():
    df = _full_dataset(3)
    snaps = _df_to_snapshots(df)
    row = SF.serve_feature_row(snaps, df.index[-1])
    assert set(F.FEATURE_COLS) <= set(row.keys())
    assert all(np.isfinite(v) for v in row.values())


def test_assemble_window_is_hourly_and_bounded():
    df = _full_dataset(400)
    now = df.index[300]
    win = SF.assemble_window(_df_to_snapshots(df.loc[:now]), now)
    assert win.index[-1] == now
    assert (win.index[-1] - win.index[0]) == pd.Timedelta(hours=F.ROLLING_WINDOW_H)
    assert (win.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()
