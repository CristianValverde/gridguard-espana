# Eficacia — paridad train/serve + proxies físicos + ensemble · Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolver la limitación de eficacia eliminando los skews train/serve con una capa de features única, añadiendo proxies físicos (inercia/ROCOF/interconexión) y fusionando LSTM+IF+CLF en el riesgo principal.

**Architecture:** Un feature store con estado: toda feature dependiente de historia se calcula con el mismo código sobre una ventana trailing (dataset en train, 168h de la tabla `snapshots` en serve). Sobre ese dataset consistente se re-entrenan los modelos y se fusionan con el LSTM como detector principal.

**Tech Stack:** Python 3.14, pandas/numpy, XGBoost+Optuna, PyTorch (LSTM AE), scikit-learn (IF), SQLite, ChromaDB + embeddings Gemini, ADK 2.3.0, pytest.

## Global Constraints

- NUNCA leer `.env`, `*.key`, `secrets.*`, `credentials.*`. Tokens vía variables de entorno.
- NUNCA hardcodear nombres de modelos Gemini — usar `config/model_loader.py` / `get_model()`.
- Shift-left: Python puro > ML > LLM.
- ADK: `google-adk>=2.3,<3`, `mcp>=1.27,<2`.
- La suite offline por defecto corre con `pytest` y debe quedar VERDE; los tests que requieren dataset/artefactos/GPU/embeddings se marcan `@pytest.mark.integration` o se auto-omiten con `skipif` (patrón ya establecido en `pytest.ini`: `addopts = -m "not integration"`, `testpaths = eval tests`).
- Commits en español, estilo conventional commits, terminados en `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Frontera de verificación:** tareas marcadas **[FRÍO]** se verifican en este worktree; **[PADRE]** requieren el dataset (`data/processed/dataset_training.parquet`), GPU y/o embeddings, y se verifican en el proyecto padre.

## File Structure

- Create `ml/features.py` — fuente única de features (funciones puras sobre DataFrame horario).
- Create `ml/serve_features.py` — ensamblado serve-time desde snapshots crudos.
- Create `tools/ensemble_scorer.py` — fusión LSTM+IF+CLF.
- Create `data/kb_seed/procedures/interconexiones_red_espanola.txt` — corpus RAG.
- Create `tests/ml/test_features.py`, `tests/ml/test_serve_features.py`, `tests/tools/test_ensemble_scorer.py`.
- Modify `ml/data/build_dataset.py` — usar `ml/features.py`; eliminar fusión ENTSO-E muerta.
- Modify `ml/train/train_classifier.py`, `ml/train/train_anomaly.py` — `FEATURE_COLS`.
- Modify `mcp_servers/ml_models_server.py` — señales LSTM/IF distintas + endpoint ensemble.
- Modify `tools/risk_scorer.py`, `agents/orchestrator.py` — cableado del score fusionado.
- Modify `ml/train/validate_models.py` — golden con contexto trailing + `validate_ensemble`.
- Modify `data/kb_seed/procedures/importacion_emergencia.txt` — mitigación por enlace.

---

### Task 1: `ml/features.py` — capa única de features  [FRÍO]

**Files:**
- Create: `ml/features.py`
- Test: `tests/ml/test_features.py`

**Interfaces:**
- Consumes: `tools.stability_proxy_calculator.compute_composite_pressure_score(balance, reserve_pct, interchange_delta, renewables_pct) -> float` (fallback cold-start).
- Produces:
  - `build_features(df: pd.DataFrame) -> pd.DataFrame` (df horario UTC con columnas raw:
    `demand_real_mw`, `nuclear_mw, hydro_mw, wind_mw, solar_pv_mw, solar_thermal_mw, gas_cc_mw, coal_mw, cogeneration_mw, pumping_mw`, `es_fr_mw, es_pt_mw, es_ma_mw`, `reserve_mw`).
  - `FEATURE_COLS: list[str]` — orden canónico de features de modelo.
  - Constantes `ROLLING_WINDOW_H=168`, `INTERCONNECT_CAPACITY_MW`, `COMPOSITE_WEIGHTS`.

- [ ] **Step 1: Escribir el test de features físicas (falla)**

```python
# tests/ml/test_features.py
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
    df = F.add_inertia_proxy(_base_df())
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
```

- [ ] **Step 2: Ejecutar y ver que falla**

Run: `python -m pytest tests/ml/test_features.py -q`
Expected: FAIL (`module ml.features has no attribute ...`).

- [ ] **Step 3: Crear `ml/features.py` con las funciones**

```python
# ml/features.py
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
    max_reserve = (demand * 0.015)
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
    Cold-start (ventana < min_periods) → fallback a la fórmula absoluta."""
    proxies = {"balance": "balance_mw", "reserve": "reserve_activated_pct",
               "interchange": "interchange_delta_mw", "renewables": "renewables_pct"}
    win = ROLLING_WINDOW_H
    composite = pd.Series(0.0, index=df.index)
    for key, col in proxies.items():
        s = _col(df, col)
        rmin = s.rolling(win, min_periods=24).min()
        rmax = s.rolling(win, min_periods=24).max()
        rng = (rmax - rmin).replace(0, np.nan)
        norm = ((s - rmin) / rng).clip(0, 1)
        composite = composite + COMPOSITE_WEIGHTS[key] * norm

    # Fallback absoluto donde el rolling no está definido (primeras 24h / huecos)
    absolute = np.vectorize(compute_composite_pressure_score)(
        _col(df, "balance_mw").fillna(0.0),
        _col(df, "reserve_activated_pct").fillna(0.0),
        _col(df, "interchange_delta_mw").fillna(0.0),
        _col(df, "renewables_pct").fillna(0.0),
    )
    df["composite_pressure"] = composite.where(composite.notna(), pd.Series(absolute, index=df.index))

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
```

- [ ] **Step 4: Ejecutar y ver que pasa**

Run: `python -m pytest tests/ml/test_features.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Test de cold-start (fallback determinista)**

```python
# añadir a tests/ml/test_features.py
def test_cold_start_composite_falls_back_to_absolute():
    from tools.stability_proxy_calculator import compute_composite_pressure_score
    df = F.build_features(_base_df(3))  # 3 filas << min_periods
    cp = df["composite_pressure"].iloc[0]
    expected = compute_composite_pressure_score(
        df["balance_mw"].iloc[0], df["reserve_activated_pct"].iloc[0],
        df["interchange_delta_mw"].iloc[0], df["renewables_pct"].iloc[0],
    )
    assert abs(cp - expected) < 1e-9
```

- [ ] **Step 6: Ejecutar y ver que pasa**

Run: `python -m pytest tests/ml/test_features.py -q`
Expected: PASS (3 tests).

- [ ] **Step 7: Commit**

```bash
git add ml/features.py tests/ml/test_features.py
git commit -m "feat(ml): capa unica de features (inercia/ROCOF/interconexion/rolling)"
```

---

### Task 2: `ml/serve_features.py` — ensamblado serve-time + test de paridad  [FRÍO]

**Files:**
- Create: `ml/serve_features.py`
- Test: `tests/ml/test_serve_features.py`

**Interfaces:**
- Consumes: `ml.features.build_features`, `ml.features.FEATURE_COLS`, `ml.features.ROLLING_WINDOW_H`.
- Produces:
  - `assemble_window(snapshots: list[dict], now: pd.Timestamp) -> pd.DataFrame`
  - `serve_feature_row(snapshots: list[dict], now: pd.Timestamp) -> dict` (claves = `FEATURE_COLS`)

- [ ] **Step 1: Test de paridad (falla)** — el corazón del diseño

```python
# tests/ml/test_serve_features.py
import numpy as np
import pandas as pd
import pytest
from ml import features as F
from ml import serve_features as SF


def _snapshots(n=200):
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    snaps = []
    for i, ts in enumerate(idx):
        r = i / n
        snaps.append({
            "timestamp": ts.isoformat(),
            "demand_mw": 30000.0,
            "generation_by_tech": {
                "nuclear_mw": 7000.0, "hydro_mw": 2000.0, "gas_cc_mw": 3000.0,
                "coal_mw": 0.0, "cogeneration_mw": 1500.0, "pumping_mw": 0.0,
                "wind_mw": 8000.0 + 4000.0 * r, "solar_pv_mw": 6000.0 + 6000.0 * r,
                "solar_thermal_mw": 500.0,
            },
            "interchange_flows": {"es_fr_mw": -1400.0 * r, "es_pt_mw": -300.0, "es_ma_mw": 50.0},
            "reserve_mw": 400.0,
        })
    return snaps, idx


def test_serve_row_matches_build_features_last_row():
    """La fila de serve == última fila de build_features sobre la misma ventana."""
    snaps, idx = _snapshots()
    now = idx[-1]
    row = SF.serve_feature_row(snaps, now)

    df_raw = SF.assemble_window(snaps, now)
    ref = F.build_features(df_raw).iloc[-1]
    for col in F.FEATURE_COLS:
        assert abs(float(row[col]) - float(ref[col])) < 1e-9, f"desajuste en {col}"
```

- [ ] **Step 2: Ejecutar y ver que falla**

Run: `python -m pytest tests/ml/test_serve_features.py -q`
Expected: FAIL (`module ml.serve_features not found`).

- [ ] **Step 3: Crear `ml/serve_features.py`**

```python
# ml/serve_features.py
"""Ensamblado serve-time del feature store con estado.

Reconstruye la ventana trailing (168h) desde snapshots crudos (DB o golden),
la lleva a índice horario UTC y aplica ml.features.build_features. Fuente-
agnóstico: acepta cualquier lista de snapshots dicts con el esquema de la tabla
`snapshots` (demand_mw, generation_by_tech, interchange_flows, reserve_mw)."""
from __future__ import annotations

import pandas as pd

from ml import features as F

_LINKS = ("es_fr_mw", "es_pt_mw", "es_ma_mw")


def assemble_window(snapshots: list[dict], now: pd.Timestamp) -> pd.DataFrame:
    if not snapshots:
        raise ValueError("assemble_window requiere al menos un snapshot")
    rows = []
    for s in snapshots:
        gen = s.get("generation_by_tech", {}) or {}
        flows = s.get("interchange_flows", {}) or {}
        row = {"timestamp": pd.Timestamp(s["timestamp"]),
               "demand_real_mw": float(s.get("demand_mw") or 0.0),
               "reserve_mw": float(s.get("reserve_mw") or 0.0)}
        for c in F.GEN_COLS:
            row[c] = float(gen.get(c) or 0.0)
        for c in _LINKS:
            row[c] = float(flows.get(c) or 0.0)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp"

    # Índice horario continuo hasta `now`, ventana de ROLLING_WINDOW_H
    now = pd.Timestamp(now)
    if now.tz is None:
        now = now.tz_localize("UTC")
    start = now - pd.Timedelta(hours=F.ROLLING_WINDOW_H)
    full_idx = pd.date_range(start=start, end=now, freq="h", tz="UTC")
    df = df.reindex(df.index.union(full_idx)).sort_index()
    df = df.reindex(full_idx, method="ffill")  # huecos pequeños → ffill
    df.index.name = "timestamp"
    return df


def serve_feature_row(snapshots: list[dict], now: pd.Timestamp) -> dict:
    df_raw = assemble_window(snapshots, now)
    feats = F.build_features(df_raw)
    last = feats.iloc[-1]
    return {c: float(last[c]) for c in F.FEATURE_COLS}
```

- [ ] **Step 4: Ejecutar y ver que pasa**

Run: `python -m pytest tests/ml/test_serve_features.py -q`
Expected: PASS. Si algún `col` desajusta, alinear `assemble_window` (reindex/ffill) hasta que la paridad sea exacta — este test es el guardián de la paridad.

- [ ] **Step 5: Test de cold-start en serve (pocos snapshots)**

```python
def test_serve_row_cold_start_does_not_crash():
    snaps, idx = _snapshots(3)
    row = SF.serve_feature_row(snaps, idx[-1])
    assert set(F.FEATURE_COLS) <= set(row.keys())
    assert all(np.isfinite(v) for v in row.values())
```

- [ ] **Step 6: Ejecutar y commit**

Run: `python -m pytest tests/ml/test_serve_features.py -q` → PASS

```bash
git add ml/serve_features.py tests/ml/test_serve_features.py
git commit -m "feat(ml): ensamblado serve-time con test de paridad train/serve"
```

---

### Task 3: `tools/ensemble_scorer.py` — fusión LSTM+IF+CLF  [FRÍO]

**Files:**
- Create: `tools/ensemble_scorer.py`
- Test: `tests/tools/test_ensemble_scorer.py`

**Interfaces:**
- Produces: `fuse(lstm_recon_norm: float, if_score: float, clf_proba: float, weights: dict | None = None) -> dict` con claves `blackout_probability` (float [0,1]), `components` (dict), `primary` (str).

- [ ] **Step 1: Tests (fallan)**

```python
# tests/tools/test_ensemble_scorer.py
import pytest
from tools.ensemble_scorer import fuse, DEFAULT_WEIGHTS, OVERRIDE_THRESHOLD, OVERRIDE_FLOOR


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
```

- [ ] **Step 2: Ejecutar y ver que falla**

Run: `python -m pytest tests/tools/test_ensemble_scorer.py -q`
Expected: FAIL (módulo no existe).

- [ ] **Step 3: Crear `tools/ensemble_scorer.py`**

```python
# tools/ensemble_scorer.py
"""Fusión ensemble del riesgo principal.

Combina tres detectores con sesgos inductivos distintos:
  - LSTM autoencoder (no supervisado, inmune a D5) → load-bearing.
  - Isolation Forest (no supervisado, anomalía puntual) → corroboración.
  - Clasificador supervisado (capado por D5) → voto secundario.

Los pesos están dominados por el LSTM POR DISEÑO (no ajustados al golden), para
mitigar el overfitting con n=1 evento S4.
"""
from __future__ import annotations

DEFAULT_WEIGHTS = {"lstm": 0.5, "if": 0.2, "clf": 0.3}
OVERRIDE_THRESHOLD = 0.9   # lstm_recon_norm ∈ [0,1]; ≥ esto = anomalía temporal fuerte
OVERRIDE_FLOOR = 0.6       # suelo de P cuando el override dispara


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def fuse(lstm_recon_norm: float, if_score: float, clf_proba: float,
         weights: dict | None = None) -> dict:
    w = weights or DEFAULT_WEIGHTS
    lstm = _clip01(lstm_recon_norm)
    isof = _clip01(if_score)
    clf = _clip01(clf_proba)

    p = w["lstm"] * lstm + w["if"] * isof + w["clf"] * clf
    p = _clip01(p)

    override = lstm >= OVERRIDE_THRESHOLD
    if override:
        p = max(p, OVERRIDE_FLOOR)

    return {
        "blackout_probability": round(p, 4),
        "primary": "lstm",
        "override_active": override,
        "components": {"lstm": round(lstm, 4), "if": round(isof, 4), "clf": round(clf, 4)},
    }
```

- [ ] **Step 4: Ejecutar y ver que pasa**

Run: `python -m pytest tests/tools/test_ensemble_scorer.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add tools/ensemble_scorer.py tests/tools/test_ensemble_scorer.py
git commit -m "feat(tools): ensemble_scorer — fusion LSTM(load-bearing)+IF+CLF"
```

---

### Task 4: Integrar `ml/features.py` en `build_dataset` + limpiar peso muerto  [PADRE]

**Files:**
- Modify: `ml/data/build_dataset.py`

**Interfaces:**
- Consumes: `ml.features.build_features`.
- Produces: `data/processed/dataset_training.parquet` con las columnas de `FEATURE_COLS` presentes.

- [ ] **Step 1:** En `build_dataset`, antes de calcular features, construir `reserve_mw` canónico desde las columnas ESIOS de reserva secundaria (valor absoluto sumado), para alimentar `ml.features`:

```python
# en compute_stability_proxies / pipeline, sustituir el bloque de proxies inline por:
reserve_cols = [c for c in dataset.columns if "secondary_reserve" in c]
dataset["reserve_mw"] = dataset[reserve_cols].abs().sum(axis=1) if reserve_cols else 0.0
```

- [ ] **Step 2:** Reemplazar las llamadas a `compute_stability_proxies(dataset)` y
`add_temporal_features(dataset)` por:

```python
from ml.features import build_features
dataset = build_features(dataset)
```

- [ ] **Step 3:** Eliminar la carga/fusión de ENTSO-E muerta (`load_entsoe` y su `concat`) y las
columnas `flow_*` (ningún modelo las usa). Mantener AEMET solo si el forecaster la usa
(`aemet_ta`, `aemet_hr`); el resto de columnas AEMET puede quedar sin dañar (no se seleccionan).

- [ ] **Step 4 [PADRE]:** Re-build y verificar columnas:

```bash
python -m ml.data.build_dataset
python -c "import pandas as pd; from ml.features import FEATURE_COLS; df=pd.read_parquet('data/processed/dataset_training.parquet'); miss=[c for c in FEATURE_COLS if c not in df.columns]; print('faltan:', miss); assert not miss"
```
Expected: `faltan: []`.

- [ ] **Step 5: Commit**

```bash
git add ml/data/build_dataset.py
git commit -m "refactor(ml): build_dataset usa ml.features; elimina fusion ENTSO-E muerta"
```

---

### Task 5: Actualizar `FEATURE_COLS` de los entrenadores  [FRÍO]

**Files:**
- Modify: `ml/train/train_classifier.py:46-56`
- Modify: `ml/train/train_anomaly.py:47-60`

**Interfaces:**
- Consumes: `ml.features.FEATURE_COLS`.

- [ ] **Step 1:** En `train_classifier.py`, sustituir la lista `FEATURE_COLS` y la construcción de
lags por import desde `ml.features` (los lags ya están en el dataset):

```python
from ml.features import FEATURE_COLS  # eliminar la lista local y LAG_FEATURES/LAG_HOURS/_add_lag_features
```
Ajustar `_prepare` para seleccionar `[c for c in FEATURE_COLS if c in df.columns]` sin recalcular lags.

- [ ] **Step 2:** En `train_anomaly.py`, sustituir la lista local `FEATURE_COLS` por el subconjunto
de detección desde `ml.features` (todas menos las temporales; el AE no necesita estacionalidad):

```python
from ml.features import FEATURE_COLS as _ALL
FEATURE_COLS = [c for c in _ALL if not c.startswith(("hour_", "dow_", "month_", "is_"))]
```

- [ ] **Step 3 [FRÍO]:** Verificar import y coherencia (sin dataset):

```bash
python -c "import ml.train.train_classifier as t; import ml.train.train_anomaly as a; print(len(t.FEATURE_COLS), len(a.FEATURE_COLS))"
```
Expected: dos enteros > 20.

- [ ] **Step 4: Commit**

```bash
git add ml/train/train_classifier.py ml/train/train_anomaly.py
git commit -m "refactor(ml): entrenadores consumen FEATURE_COLS de ml.features"
```

---

### Task 6: Re-entrenar IF + LSTM + clasificador  [PADRE]

**Files:** (artefactos) `ml/models/candidates/anomaly_detector.pkl`, `lstm_ae.pth`, `blackout_classifier.pkl`

- [ ] **Step 1 [PADRE]:** Re-entrenar sobre el dataset nuevo (GPU):

```bash
python -m ml.train.train_anomaly            # IF + LSTM → candidates/
python -m ml.train.train_classifier         # XGBoost + Optuna → candidates/
```
Expected: los tres artefactos en `ml/models/candidates/`; logs con AUC-ROC LOEO del clasificador y umbral del LSTM.

- [ ] **Step 2 [PADRE]:** Registrar en el ledger el AUC nuevo (esperado ~0.73 recuperado por el rolling replicable). No promover aún: la promoción ocurre tras `validate_models` (Task 10).

- [ ] **Step 3: Commit** (solo si se versionan métricas/notas; los artefactos están gitignored)

```bash
git commit --allow-empty -m "chore(ml): re-entrenamiento IF/LSTM/CLF sobre features nuevas (candidates)"
```

---

### Task 7: Re-derivar umbrales Markov  [PADRE]

**Files:** (artefacto) `ml/models/markov_state.npz`

- [ ] **Step 1 [PADRE]:** Re-fit sobre el dataset nuevo (usa `composite_pressure` rolling):

```bash
python -m ml.grid_state_tracker
```
Expected: `markov_state.npz` regenerado; log con la distribución de estados S0–S4 y los umbrales P90/P97 nuevos.

- [ ] **Step 2 [PADRE]:** Verificar que los umbrales bajaron respecto a la distribución absoluta (el rolling reescala): comparar el log con el valor previo (P90≈0.713). Documentar en el ledger.

- [ ] **Step 3: Commit**

```bash
git commit --allow-empty -m "chore(ml): re-fit Markov sobre composite rolling (umbrales nuevos)"
```

---

### Task 8: `ml_models_server` — señales LSTM/IF distintas + endpoint ensemble  [FRÍO]

**Files:**
- Modify: `mcp_servers/ml_models_server.py`
- Test: `tests/mcp_servers/test_ensemble_endpoint.py` (Create)

**Interfaces:**
- Consumes: `tools.ensemble_scorer.fuse`.
- Produces: tool MCP `get_ensemble_score(features_json: str, window_json: str) -> dict` que devuelve
  `{"blackout_probability": float, "components": {...}, "primary": "lstm"}`.

- [ ] **Step 1:** En `get_anomaly_score`, dejar de usar `elif` LSTM/IF: calcular AMBOS cuando estén
disponibles y devolver `{"if_score": float, "lstm_recon_norm": float}` además del score legacy.

- [ ] **Step 2:** Añadir el tool de ensemble que orquesta las tres señales:

```python
@mcp.tool()
def get_ensemble_score(features_json: str, window_json: str) -> dict:
    """Fusiona LSTM+IF+CLF en P(blackout). features_json → clasificador;
    window_json → secuencia para IF/LSTM."""
    from tools.ensemble_scorer import fuse
    anomaly = get_anomaly_score(window_json)          # {"if_score","lstm_recon_norm",...}
    clf = get_blackout_probability(features_json)      # {"blackout_probability",...}
    return fuse(
        lstm_recon_norm=float(anomaly.get("lstm_recon_norm", 0.0)),
        if_score=float(anomaly.get("if_score", 0.0)),
        clf_proba=float(clf.get("blackout_probability", 0.0)),
    )
```

- [ ] **Step 2b:** Test con modelos ausentes (frío) — el endpoint degrada sin crashear:

```python
# tests/mcp_servers/test_ensemble_endpoint.py
import json
from mcp_servers import ml_models_server as S


def test_ensemble_degrades_without_models(monkeypatch):
    monkeypatch.setattr(S, "get_anomaly_score", lambda _w: {"if_score": 0.0, "lstm_recon_norm": 0.0})
    monkeypatch.setattr(S, "get_blackout_probability", lambda _f: {"blackout_probability": 0.0})
    out = S.get_ensemble_score("{}", "{}")
    assert 0.0 <= out["blackout_probability"] <= 1.0
    assert out["primary"] == "lstm"
```

- [ ] **Step 3 [FRÍO]:** Ejecutar:

Run: `python -m pytest tests/mcp_servers/test_ensemble_endpoint.py -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add mcp_servers/ml_models_server.py tests/mcp_servers/test_ensemble_endpoint.py
git commit -m "feat(mcp): endpoint get_ensemble_score + senales LSTM/IF distintas"
```

---

### Task 9: Cablear el score fusionado en `risk_scorer` + `orchestrator`  [FRÍO]

**Files:**
- Modify: `tools/risk_scorer.py`
- Modify: `agents/orchestrator.py` (instrucción + whitelist si aplica)
- Test: `tests/tools/test_risk_scorer_ensemble.py` (Create)

**Interfaces:**
- `risk_scorer(snapshot, ml_score, baseline=None)` — `ml_score` pasa a ser el `blackout_probability`
  fusionado. La firma no cambia (compatibilidad), solo su origen.

- [ ] **Step 1:** Test: `risk_scorer` con el score fusionado propaga la P y sube de nivel:

```python
# tests/tools/test_risk_scorer_ensemble.py
from tools.risk_scorer import risk_scorer

_SNAP = {"demand_mw": 30000, "generation_by_tech": {"wind_mw": 14000, "solar_pv_mw": 13000},
         "interchange_flows": {"es_fr_mw": -1800}, "reserve_mw": 300}


def test_fused_score_drives_blackout_probability():
    out = risk_scorer(_SNAP, ml_score=0.82)
    assert abs(out["blackout_probability"] - 0.82) < 1e-3
    assert out["level"] in ("HIGH", "CRITICAL")
```

- [ ] **Step 2 [FRÍO]:** Ejecutar (debe pasar ya; `risk_scorer` no cambia su lógica):

Run: `python -m pytest tests/tools/test_risk_scorer_ensemble.py -q`
Expected: PASS.

- [ ] **Step 3:** En `agents/orchestrator.py`, actualizar la instrucción del paso ML: llamar a
`get_ensemble_score` (en vez de solo `get_blackout_probability`) y pasar su `blackout_probability`
como `ml_score` a `risk_scorer`. Añadir `get_ensemble_score` a la whitelist del orchestrator en
`guardrails/hooks.py::_ALLOWED_TOOLS["gridguard_orchestrator"]`.

- [ ] **Step 4 [FRÍO]:** Verificar import de agentes y suite de guardrails:

Run: `python -m pytest tests/agents eval/test_outputs.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/risk_scorer.py agents/orchestrator.py guardrails/hooks.py tests/tools/test_risk_scorer_ensemble.py
git commit -m "feat(agents): orchestrator usa get_ensemble_score -> risk_scorer"
```

---

### Task 10: `validate_models` — golden con contexto trailing + `validate_ensemble`  [PADRE]

**Files:**
- Modify: `ml/train/validate_models.py`

**Interfaces:**
- Consumes: `ml.serve_features.serve_feature_row`, `tools.ensemble_scorer.fuse`.

- [ ] **Step 1:** Sustituir `_snapshot_to_row` (que ponía lags a 0 y usaba composite absoluto) por el
uso de `serve_feature_row`, alimentado con la **semana previa real** extraída del dataset ESIOS:

```python
def _golden_context_snapshots(scenario, dataset_path="data/processed/dataset_training.parquet"):
    """Devuelve la semana previa (168h) al primer snapshot del golden como snapshots crudos,
    reconstruidos del dataset (misma fuente ESIOS) + los snapshots del golden."""
    # cargar dataset, recortar [t0-168h, t0), convertir filas a dict-snapshot, concatenar golden
```

- [ ] **Step 2:** Añadir `validate_ensemble(df_blackout, df_normal)` que calcula el score fusionado
por snapshot y verifica: (a) media(golden) > media(normal); (b) monotonía no decreciente hacia el
onset en los últimos 3 snapshots.

- [ ] **Step 3 [PADRE]:** Ejecutar la validación completa:

```bash
python -m ml.train.validate_models
```
Expected: LSTM PASS; `ensemble` PASS (golden>normal y creciente). Documentar honestamente en el
ledger si el CLF sigue débil (D5) — el ensemble lo sostiene el LSTM.

- [ ] **Step 4 [PADRE]:** Si PASS, promover candidatos a `ml/models/` (conservando los antiguos como
backup) y re-ejecutar `validate_models` apuntando a producción.

- [ ] **Step 5: Commit**

```bash
git add ml/train/validate_models.py
git commit -m "feat(ml): validate_ensemble + golden con contexto trailing (fin skew validacion)"
```

---

### Task 11: RAG — contexto de interconexiones  [PADRE]

**Files:**
- Create: `data/kb_seed/procedures/interconexiones_red_espanola.txt`
- Modify: `data/kb_seed/procedures/importacion_emergencia.txt`
- Test: `tests/mcp_servers/test_kb_interconnexiones.py` (Create, `@pytest.mark.integration`)

- [ ] **Step 1:** Crear el documento de corpus (texto plano, estilo de los `.txt` existentes) con
secciones: carácter de isla eléctrica de la península ibérica; ES↔FR (único enlace AC síncrono
continental, salvavidas de inercia, NTC ~2800 MW); ES↔PT (coupling intra-ibérico MIBEL, co-víctima
el 28-abr); ES↔MA (HVDC ~900 MW, rol en restauración); papel de cada enlace en el apagón del
28-abr-2025. Ubicarlo en `procedures/` (reachable por `cross_border` vía `get_mitigation_procedure`).

- [ ] **Step 2:** Enriquecer `importacion_emergencia.txt` con mitigación por enlace (import de
emergencia desde FR; soporte DC de MA).

- [ ] **Step 3 [PADRE]:** Re-seed (idempotente):

```bash
python -m scripts.seed_knowledge_base
```
Expected: chunks nuevos indexados en la colección `procedures`.

- [ ] **Step 4 [PADRE]:** Smoke test de recuperación:

```python
# tests/mcp_servers/test_kb_interconnexiones.py
import pytest
from mcp_servers.knowledge_base_server import get_mitigation_procedure


@pytest.mark.integration
def test_interconnection_context_retrievable():
    res = get_mitigation_procedure("capacidad de la interconexion Espana-Francia")
    txt = str(res).lower()
    assert "francia" in txt or "es_fr" in txt or "interconexion" in txt
```

Run: `python -m pytest tests/mcp_servers/test_kb_interconnexiones.py -q -m integration`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add data/kb_seed/procedures/interconexiones_red_espanola.txt data/kb_seed/procedures/importacion_emergencia.txt tests/mcp_servers/test_kb_interconnexiones.py
git commit -m "feat(kb): contexto RAG de interconexiones ES-FR/PT/MA + re-seed"
```

---

## Cierre

- [ ] **Verificación final [FRÍO]:** `python -m pytest -q` → verde (nuevos tests incluidos; los
  `integration`/dataset se omiten/deseleccionan).
- [ ] **Verificación final [PADRE]:** `validate_models` verde con artefactos promovidos; ledger
  actualizado con AUC nuevo, distribución Markov nueva y conclusión honesta del ensemble.
- [ ] Actualizar `.superpowers/sdd/progress.md` y `ARCHITECTURE.md` (nueva capa de features,
  ensemble, features de interconexión).

## Orden y dependencias

```
1 (features) ─┬─> 2 (serve) ──> 8 (mcp) ──> 9 (wiring)
              ├─> 3 (ensemble) ┘
              └─> 4 (build_dataset) ──> 5 (trainers) ──> 6 (retrain) ──> 7 (markov) ──> 10 (validate)
11 (RAG) es independiente (puede ir en paralelo).
```
Tareas 1,2,3,5,8,9,11(código) son **[FRÍO]**. Tareas 4,6,7,10 y los re-seed/validación son **[PADRE]**.
