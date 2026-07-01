import numpy as np
import pandas as pd
import pytest
from ml import features as F


def _base_df(n=200):
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    rng = np.linspace(0, 1, n)
    df = pd.DataFrame(index=idx)
    df.index.name = "timestamp"
    df["demand_real_mw"] = 30000.0
    df["nuclear_mw"] = 7000.0
    df["hydro_mw"] = 2000.0
    df["gas_cc_mw"] = 3000.0
    df["coal_mw"] = 0.0
    df["cogeneration_mw"] = 1500.0
    df["pumping_mw"] = 0.0
    df["wind_mw"] = 8000.0 + 4000.0 * rng
    df["solar_pv_mw"] = 6000.0 + 6000.0 * rng
    df["solar_thermal_mw"] = 500.0
    df["es_fr_mw"] = -1400.0 * rng          # export creciente hacia FR
    df["es_pt_mw"] = -300.0
    df["es_ma_mw"] = 50.0
    df["reserve_mw"] = 400.0
    return df


def test_inertia_proxy_ranges_and_direction():
    df = F.add_inertia_proxy(F.add_base_proxies(_base_df()))
    nsp = df["non_synchronous_penetration"]
    assert (nsp >= 0).all() and (nsp <= 1).all()
    # más renovable al final => mayor penetración no-síncrona
    assert nsp.iloc[-1] > nsp.iloc[0]


def test_interconnect_stress_per_link_and_total():
    df = F.add_interconnect_stress(_base_df())
    assert {"stress_es_fr", "stress_es_pt", "stress_es_ma", "interconnect_stress_total"} <= set(df.columns)
    assert (df["stress_es_fr"] >= 0).all() and (df["stress_es_fr"] <= 1).all()
    # el estrés FR crece con el flujo
    assert df["stress_es_fr"].iloc[-1] > df["stress_es_fr"].iloc[0]


def test_cold_start_composite_falls_back_to_absolute():
    from tools.stability_proxy_calculator import compute_composite_pressure_score
    df = F.build_features(_base_df(3))  # 3 filas << min_periods
    cp = df["composite_pressure"].iloc[0]
    expected = compute_composite_pressure_score(
        df["balance_mw"].iloc[0], df["reserve_activated_pct"].iloc[0],
        df["interchange_delta_mw"].iloc[0], df["renewables_pct"].iloc[0],
    )
    assert abs(cp - expected) < 1e-9


def test_build_features_produces_all_feature_cols():
    df = F.build_features(_base_df())
    missing = [c for c in F.FEATURE_COLS if c not in df.columns]
    assert not missing, f"faltan columnas: {missing}"
