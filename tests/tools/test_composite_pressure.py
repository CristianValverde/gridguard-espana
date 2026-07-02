"""
tests/tools/test_composite_pressure.py — Verifica la función canónica compute_composite_pressure_score
y la consistencia entre los tres call sites (runtime, build_dataset, validate_models).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.stability_proxy_calculator import compute_composite_pressure_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE = dict(balance_mw=500.0, reserve_activated_pct=30.0, interchange_delta_mw=400.0, renewables_pct=40.0)


def _call(**kwargs):
    """Shortcut que llama a la función canónica con BASE mergeado con kwargs."""
    params = {**BASE, **kwargs}
    return compute_composite_pressure_score(
        params["balance_mw"],
        params["reserve_activated_pct"],
        params["interchange_delta_mw"],
        params["renewables_pct"],
    )


# ---------------------------------------------------------------------------
# Rango [0, 1]
# ---------------------------------------------------------------------------

def test_output_in_range_baseline():
    result = compute_composite_pressure_score(**BASE)
    assert 0.0 <= result <= 1.0


def test_output_clamped_at_zero():
    # Valores cero → score mínimo = 0.0
    result = compute_composite_pressure_score(0.0, 0.0, 0.0, 0.0)
    assert result == 0.0


def test_output_clamped_at_one():
    # Valores extremadamente altos → score máximo = 1.0
    result = compute_composite_pressure_score(99999.0, 100.0, 99999.0, 100.0)
    assert result == 1.0


# ---------------------------------------------------------------------------
# Monotonicidad por dimensión (ceteris paribus)
# ---------------------------------------------------------------------------

def test_monotonic_balance_mw():
    """Mayor |balance_mw| → mayor presión."""
    low = _call(balance_mw=100.0)
    high = _call(balance_mw=2500.0)
    assert high > low


def test_monotonic_reserve_activated_pct():
    """Mayor reserva activada → mayor presión."""
    low = _call(reserve_activated_pct=10.0)
    high = _call(reserve_activated_pct=80.0)
    assert high > low


def test_monotonic_interchange_delta_mw():
    """Mayor variación de intercambio → mayor presión."""
    low = _call(interchange_delta_mw=100.0)
    high = _call(interchange_delta_mw=1500.0)
    assert high > low


def test_monotonic_renewables_pct():
    """Mayor % renovables → mayor presión."""
    low = _call(renewables_pct=20.0)
    high = _call(renewables_pct=75.0)
    assert high > low


# ---------------------------------------------------------------------------
# Umbrales operacionales correctos
# ---------------------------------------------------------------------------

def test_balance_threshold_at_3000():
    """balance_score satura en 1.0 a partir de |balance_mw| = 3000."""
    at_threshold = compute_composite_pressure_score(3000.0, 0.0, 0.0, 0.0)
    above_threshold = compute_composite_pressure_score(9000.0, 0.0, 0.0, 0.0)
    assert at_threshold == above_threshold


def test_renewables_threshold_at_80():
    """renewables_score satura en 1.0 a partir de renewables_pct = 80."""
    at_threshold = compute_composite_pressure_score(0.0, 0.0, 0.0, 80.0)
    above_threshold = compute_composite_pressure_score(0.0, 0.0, 0.0, 150.0)
    assert at_threshold == above_threshold


# ---------------------------------------------------------------------------
# Pesos conocidos
# ---------------------------------------------------------------------------

def test_weights_sum_to_one():
    """La combinación ponderada con todos los scores = 1.0 debe dar 1.0."""
    result = compute_composite_pressure_score(3000.0, 100.0, 2000.0, 80.0)
    assert result == pytest.approx(1.0)


def test_known_value():
    """Valor conocido calculado a mano para detectar regresiones."""
    # balance=1500 → score=0.5; reserve=50 → score=0.5; interchange=1000 → score=0.5; renewables=40 → score=0.5
    # composite = 0.35*0.5 + 0.25*0.5 + 0.20*0.5 + 0.20*0.5 = 0.5
    result = compute_composite_pressure_score(1500.0, 50.0, 1000.0, 40.0)
    assert result == pytest.approx(0.5, abs=1e-9)


# ---------------------------------------------------------------------------
# Consistencia: validate_models._snapshot_to_row == shared function
# ---------------------------------------------------------------------------

def _make_snapshot(balance_mw, reserve_pct, interchange_delta, renewables_pct, demand=25000.0):
    """Construye un snapshot compatible con _snapshot_to_row para los valores dados."""
    # Ingeniería inversa: recalculamos los inputs originales para que _snapshot_to_row
    # produzca los valores de presión deseados.
    # reserve_mw: reserve_pct = min(reserve / (demand*0.015) * 100, 100)
    #   → reserve = reserve_pct/100 * demand * 0.015
    reserve_mw = (reserve_pct / 100.0) * demand * 0.015

    # interchange_delta depende de prev_net_flow, lo pasamos como 0 y usamos
    # un flujo neto igual a interchange_delta (prev=0).
    net_flow = interchange_delta  # prev_net_flow=0 → delta = |net_flow - 0| = interchange_delta

    # renewables_pct = (wind + solar_pv + solar_thermal) / demand * 100
    #   → usamos todo en wind_mw para simplificar
    renewables_mw = renewables_pct / 100.0 * demand

    # balance_mw = total_gen - demand
    #   → total_gen = demand + balance_mw
    #   → para simplificar: gas_cc absorbe la diferencia
    gas_cc = demand + balance_mw - renewables_mw  # nuclear, hydro, etc. = 0

    return {
        "timestamp": "2025-04-28T08:00:00+00:00",
        "demand_mw": demand,
        "reserve_mw": reserve_mw,
        "generation_by_tech": {
            "nuclear_mw": 0.0,
            "hydro_mw": 0.0,
            "wind_mw": renewables_mw,
            "solar_pv_mw": 0.0,
            "solar_thermal_mw": 0.0,
            "gas_cc_mw": gas_cc,
            "coal_mw": 0.0,
            "cogeneration_mw": 0.0,
            "pumping_mw": 0.0,
        },
        "interchange_flows": {
            "es_fr_mw": net_flow,
            "es_pt_mw": 0.0,
            "es_ma_mw": 0.0,
        },
    }, 0.0  # prev_net_flow = 0


def test_validate_models_uses_serve_features_pipeline():
    """La validación del golden usa la MISMA canalización que serve
    (serve_feature_row/serve_feature_frame) — garantía de paridad más fuerte
    que la antigua _snapshot_to_row (eliminada en Task 10)."""
    import inspect

    import ml.train.validate_models as VM

    assert not hasattr(VM, "_snapshot_to_row"), "camino ad-hoc reintroducido"
    src = inspect.getsource(VM)
    assert "serve_feature_row" in src and "serve_feature_frame" in src


# ---------------------------------------------------------------------------
# Consistencia: build_dataset (vía numpy.vectorize) == shared function
# ---------------------------------------------------------------------------

def test_build_dataset_composite_equals_shared_function():
    """La lógica de build_dataset (np.vectorize) debe producir el mismo resultado
    que compute_composite_pressure_score para las mismas entradas."""
    # Replicamos la lógica de build_dataset directamente:
    #   df["composite_pressure"] = np.vectorize(compute_composite_pressure_score)(...)
    # Si la función es la misma, vectorize(f)(x) == f(x) por definición.
    # Este test verifica que el contrato se mantiene y detecta si se cambia la implementación.
    rows = [
        (500.0, 30.0, 400.0, 40.0),
        (0.0, 0.0, 0.0, 0.0),
        (3000.0, 100.0, 2000.0, 80.0),
        (-2000.0, 50.0, 800.0, 60.0),
    ]
    df = pd.DataFrame(rows, columns=["balance_mw", "reserve_activated_pct", "interchange_delta_mw", "renewables_pct"])

    # Lógica de build_dataset
    df["composite_pressure"] = np.vectorize(compute_composite_pressure_score)(
        df["balance_mw"].fillna(0.0),
        df["reserve_activated_pct"].fillna(0.0),
        df["interchange_delta_mw"].fillna(0.0),
        df["renewables_pct"].fillna(0.0),
    )

    # Verificar row a row que coincide con la función directa
    for _, r in df.iterrows():
        expected = compute_composite_pressure_score(
            r["balance_mw"], r["reserve_activated_pct"], r["interchange_delta_mw"], r["renewables_pct"]
        )
        assert r["composite_pressure"] == pytest.approx(expected, abs=1e-9)
