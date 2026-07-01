"""Fuente única de features de GridGuard.

Usada por build_dataset (train), serve_features (serve) y validate_models
(validación). La paridad train/serve queda garantizada por construcción: toda
feature dependiente de historia se calcula aquí, sobre un DataFrame horario UTC.

Contrato de entrada de build_features(df): índice DatetimeIndex horario UTC y
columnas raw:
  demand_real_mw, nuclear_mw, hydro_mw, wind_mw, solar_pv_mw, solar_thermal_mw,
  gas_cc_mw, coal_mw, cogeneration_mw, pumping_mw, es_fr_mw, es_pt_mw, es_ma_mw,
  reserve_mw
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from tools.stability_proxy_calculator import compute_composite_pressure_score

ROLLING_WINDOW_H = 168
COMPOSITE_WEIGHTS = {"balance": 0.35, "reserve": 0.25, "interchange": 0.20, "renewables": 0.20}
# NTC típicos (MW). Verificar contra valores publicados de REE/ENTSO-E.
INTERCONNECT_CAPACITY_MW = {"es_fr": 2800.0, "es_pt": 3000.0, "es_ma": 900.0}

GEN_COLS = ["nuclear_mw", "hydro_mw", "wind_mw", "solar_pv_mw", "solar_thermal_mw",
            "gas_cc_mw", "coal_mw", "cogeneration_mw", "pumping_mw"]
RENEWABLE_COLS = ["wind_mw", "solar_pv_mw", "solar_thermal_mw"]
SYNC_COLS = ["nuclear_mw", "hydro_mw", "gas_cc_mw", "coal_mw", "cogeneration_mw"]
LINK_COLS = {"es_fr": "es_fr_mw", "es_pt": "es_pt_mw", "es_ma": "es_ma_mw"}


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name] if name in df.columns else pd.Series(0.0, index=df.index)


def add_base_proxies(df: pd.DataFrame) -> pd.DataFrame:
    total_gen = sum(_col(df, c) for c in GEN_COLS)
    df["balance_mw"] = total_gen - _col(df, "demand_real_mw")
    demand = _col(df, "demand_real_mw").replace(0, np.nan)
    max_reserve = demand * 0.015
    df["reserve_activated_pct"] = (_col(df, "reserve_mw").abs() / max_reserve * 100).clip(0, 100).fillna(0.0)
    net_flow = sum(_col(df, c) for c in LINK_COLS.values())
    df["interchange_delta_mw"] = net_flow.diff(1).abs().fillna(0.0)
    renew = sum(_col(df, c) for c in RENEWABLE_COLS)
    df["renewables_pct"] = (renew / demand * 100).clip(0, 100).fillna(0.0)
    return df


def add_inertia_proxy(df: pd.DataFrame) -> pd.DataFrame:
    total_gen = sum(_col(df, c) for c in GEN_COLS).replace(0, np.nan)
    renew = sum(_col(df, c) for c in RENEWABLE_COLS)
    df["synchronous_gen_mw"] = sum(_col(df, c) for c in SYNC_COLS)
    df["non_synchronous_penetration"] = (renew / total_gen).clip(0, 1).fillna(0.0)
    return df


def add_interconnect_stress(df: pd.DataFrame) -> pd.DataFrame:
    total_cap = sum(INTERCONNECT_CAPACITY_MW.values())
    net = pd.Series(0.0, index=df.index)
    for key, col in LINK_COLS.items():
        flow = _col(df, col)
        df[f"stress_{key}"] = (flow.abs() / INTERCONNECT_CAPACITY_MW[key]).clip(0, 1)
        net = net + flow
    df["interconnect_stress_total"] = (net.abs() / total_cap).clip(0, 1)
    return df


def add_deltas(df: pd.DataFrame) -> pd.DataFrame:
    df["balance_delta_1h"] = _col(df, "balance_mw").diff(1).fillna(0.0)
    df["nsp_delta_1h"] = _col(df, "non_synchronous_penetration").diff(1).fillna(0.0)
    df["renewables_delta_1h"] = _col(df, "renewables_pct").diff(1).fillna(0.0)
    return df


def add_volatility(df: pd.DataFrame, windows=(6, 24)) -> pd.DataFrame:
    for w in windows:
        df[f"balance_std_{w}h"] = _col(df, "balance_mw").rolling(w, min_periods=2).std().fillna(0.0)
        df[f"interchange_std_{w}h"] = _col(df, "interchange_delta_mw").rolling(w, min_periods=2).std().fillna(0.0)
        df[f"nsp_std_{w}h"] = _col(df, "non_synchronous_penetration").rolling(w, min_periods=2).std().fillna(0.0)
    return df


def add_rolling_norm(df: pd.DataFrame) -> pd.DataFrame:
    """composite_pressure = min-max trailing 7d por proxy + suma ponderada.

    Cold-start (ventana < min_periods) → fallback a la fórmula absoluta.
    """
    proxies = {"balance": "balance_mw", "reserve": "reserve_activated_pct",
               "interchange": "interchange_delta_mw", "renewables": "renewables_pct"}
    win = ROLLING_WINDOW_H
    composite = pd.Series(0.0, index=df.index)
    valid = pd.Series(True, index=df.index)
    for key, col in proxies.items():
        s = _col(df, col)
        rmin = s.rolling(win, min_periods=24).min()
        rmax = s.rolling(win, min_periods=24).max()
        rng = (rmax - rmin).replace(0, np.nan)
        norm = ((s - rmin) / rng).clip(0, 1)
        valid = valid & norm.notna()
        composite = composite + COMPOSITE_WEIGHTS[key] * norm.fillna(0.0)

    # Fallback absoluto donde el rolling no está definido (primeras 24h / huecos)
    absolute = np.vectorize(compute_composite_pressure_score)(
        _col(df, "balance_mw").fillna(0.0).to_numpy(),
        _col(df, "reserve_activated_pct").fillna(0.0).to_numpy(),
        _col(df, "interchange_delta_mw").fillna(0.0).to_numpy(),
        _col(df, "renewables_pct").fillna(0.0).to_numpy(),
    )
    absolute = pd.Series(absolute, index=df.index)
    df["composite_pressure"] = composite.where(valid, absolute)

    # z-scores trailing 7d (desviación del normal reciente)
    for name, col in (("balance_z_7d", "balance_mw"), ("renewables_z_7d", "renewables_pct")):
        s = _col(df, col)
        mean = s.rolling(win, min_periods=24).mean()
        std = s.rolling(win, min_periods=24).std().replace(0, np.nan)
        df[name] = ((s - mean) / std).fillna(0.0)
    return df


def add_lags(df: pd.DataFrame, lags=(1, 3, 6)) -> pd.DataFrame:
    for col in ("composite_pressure", "balance_mw", "non_synchronous_penetration"):
        for lag in lags:
            df[f"{col}_lag_{lag}h"] = _col(df, col).shift(lag).fillna(0.0)
    return df


def add_temporal(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    df["hour_sin"] = np.sin(2 * math.pi * idx.hour / 24)
    df["hour_cos"] = np.cos(2 * math.pi * idx.hour / 24)
    df["dow_sin"] = np.sin(2 * math.pi * idx.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * math.pi * idx.dayofweek / 7)
    df["month_sin"] = np.sin(2 * math.pi * idx.month / 12)
    df["month_cos"] = np.cos(2 * math.pi * idx.month / 12)
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    try:
        import holidays as hols
        es = hols.Spain(years=range(idx.year.min(), idx.year.max() + 1))
        holiday_dates = pd.DatetimeIndex(list(es.keys())).tz_localize("UTC")
        df["is_holiday_es"] = idx.normalize().isin(holiday_dates).astype(int)
    except ImportError:
        df["is_holiday_es"] = 0
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = add_base_proxies(df)
    df = add_inertia_proxy(df)
    df = add_interconnect_stress(df)
    df = add_deltas(df)
    df = add_volatility(df)
    df = add_rolling_norm(df)
    df = add_lags(df)
    df = add_temporal(df)
    return df


FEATURE_COLS: list[str] = [
    # base
    "demand_real_mw", "balance_mw", "reserve_activated_pct", "interchange_delta_mw",
    "renewables_pct", "composite_pressure",
    # física
    "non_synchronous_penetration", "synchronous_gen_mw",
    "balance_delta_1h", "nsp_delta_1h", "renewables_delta_1h",
    "balance_std_6h", "balance_std_24h", "interchange_std_6h", "interchange_std_24h",
    "nsp_std_6h", "nsp_std_24h",
    "balance_z_7d", "renewables_z_7d",
    # interconexión
    "stress_es_fr", "stress_es_pt", "stress_es_ma", "interconnect_stress_total",
    # lags
    "composite_pressure_lag_1h", "composite_pressure_lag_3h", "composite_pressure_lag_6h",
    "balance_mw_lag_1h", "balance_mw_lag_3h", "balance_mw_lag_6h",
    "non_synchronous_penetration_lag_1h", "non_synchronous_penetration_lag_3h",
    "non_synchronous_penetration_lag_6h",
    # temporales
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_weekend", "is_holiday_es",
]
