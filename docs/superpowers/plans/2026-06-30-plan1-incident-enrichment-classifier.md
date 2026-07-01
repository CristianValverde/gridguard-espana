# Plan 1 — Enriquecimiento de incidentes + re-entrenamiento del clasificador

> **Para trabajadores agénticos:** SUB-SKILL REQUERIDO: usa `superpowers:subagent-driven-development`
> (recomendado) o `superpowers:executing-plans` para implementar este plan tarea a tarea. Los pasos usan
> checkbox (`- [ ]`). Lee el [índice maestro](2026-06-30-gridguard-incident-markov-agents-INDEX.md) primero:
> aloja los constraints globales, el registro de incidentes verificado y las decisiones D1–D6.

**Objetivo:** sustituir el etiquetado roto del clasificador (timestamp del apagón desplazado 2,5 h, 1 solo
evento en rango, AUC=0 en todos los folds) por un registro de incidentes verificado S0–S4, etiquetas con
ventanas graduadas, validación Leave-One-Event-Out, y re-entrenar el clasificador dejando `validate_models`
en verde.

**Arquitectura:** un módulo nuevo `ml/incidents.py` se convierte en la única fuente de verdad de incidentes y
estados, consumido por `train_classifier.py` (este plan) y por la capa de Markov (Plan 2). El clasificador
pasa a ser binario "evento mayor inminente" con `sample_weight` por severidad (D3) y validación
Leave-One-Event-Out (D4).

**Tech stack:** XGBoost 3.3.0 (GPU/CUDA), scikit-learn 1.8.0, optuna 4.9.0, pandas, joblib, pytest.

## Constraints globales

Ver [índice maestro § Constraints globales](2026-06-30-gridguard-incident-markov-agents-INDEX.md). Resumen
operativo para este plan: UTC en todo; shift-left (la derivación de estados es Python puro); nunca leer
`.env`; no hardcodear modelos Gemini (este plan no toca LLM); TDD + commit por tarea.

---

## Estructura de ficheros

- **Crear** `ml/incidents.py` — registro de incidentes S0–S4 verificado + helpers de etiquetado y derivación
  de estados. Única fuente de verdad, compartida con Plan 2.
- **Crear** `tests/ml/test_incidents.py` — tests del registro, etiquetado graduado y derivación de estados.
- **Modificar** `ml/train/train_classifier.py` — eliminar `KNOWN_INCIDENTS`/`build_labels`; consumir
  `ml/incidents.py`; añadir `leave_one_event_out_splits`; pasar `sample_weight`.
- **Crear** `tests/ml/test_train_classifier.py` — tests de splits y forma del bundle.
- **Modificar** `ml/train/validate_models.py` — unificar nombre forecaster (D6) + leer el bundle nuevo.
- **Modificar** `ml/retrain_pipeline.py:43` — `load_forecaster.pkl` → `load_forecaster.pkl` (D6).

---

### Task 1: Módulo `ml/incidents.py` con el registro verificado y timestamp corregido

**Files:**
- Create: `ml/incidents.py`
- Test: `tests/ml/test_incidents.py`

**Interfaces:**
- Produce: `S0_NORMAL=0, S1_STRESS=1, S2_NOTABLE=2, S3_MAJOR=3, S4_BLACKOUT=4` (constantes int);
  `DOCUMENTED_INCIDENTS: list[dict]`; `documented_incidents_utc() -> list[dict]` (cada dict con claves
  `ts: pd.Timestamp[UTC]`, `severity: int`, `label: str`, `window_h: int`, opcional `onset: str`).

- [ ] **Step 1: Escribir el test que falla**

```python
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
```

- [ ] **Step 2: Ejecutar el test para verificar que falla**

Run: `python -m pytest tests/ml/test_incidents.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'ml.incidents'`.

- [ ] **Step 3: Implementar `ml/incidents.py`**

```python
"""
ml/incidents.py — Única fuente de verdad de incidentes de la red española y de la
taxonomía de estados S0–S4. Consumido por ml/train/train_classifier.py (Plan 1) y por
la capa de Markov ml/grid_state_tracker.py (Plan 2).

Todas las horas en UTC. Ver docs/superpowers/plans/2026-06-30-...-INDEX.md (registro verificado).
"""

from __future__ import annotations

import pandas as pd

# ── Taxonomía de estados (D1) ───────────────────────────────────────────────
S0_NORMAL = 0
S1_STRESS = 1
S2_NOTABLE = 2
S3_MAJOR = 3
S4_BLACKOUT = 4

STATE_LABELS = {
    S0_NORMAL: "Operación normal",
    S1_STRESS: "Estrés de red",
    S2_NOTABLE: "Incidente notable",
    S3_MAJOR: "Incidente mayor",
    S4_BLACKOUT: "Apagón total",
}

# ── Eventos discretos documentados (curados, UTC). Ver INDEX para fuentes. ───
# window_h = horas previas etiquetadas como positivo (graduado: S4 > S3).
DOCUMENTED_INCIDENTS: list[dict] = [
    {
        "ts": "2021-01-08 13:04",
        "severity": S3_MAJOR,
        "label": "Separación área síncrona continental europea",
        "window_h": 3,
    },
    {
        "ts": "2021-07-24 14:35",
        "severity": S3_MAJOR,
        "label": "Desconexión península-Europa (Hernani-Argia)",
        "window_h": 3,
    },
    {
        "ts": "2025-04-28 10:33",
        "severity": S4_BLACKOUT,
        "label": "Apagón total ibérico",
        "window_h": 6,
        "onset": "2025-04-28 10:03",  # precursor: oscilación 0,6 Hz
    },
]


def documented_incidents_utc() -> list[dict]:
    """Devuelve los incidentes con el timestamp convertido a pd.Timestamp tz-aware (UTC)."""
    out = []
    for inc in DOCUMENTED_INCIDENTS:
        item = dict(inc)
        item["ts"] = pd.Timestamp(inc["ts"], tz="UTC")
        out.append(item)
    return out
```

- [ ] **Step 4: Ejecutar el test para verificar que pasa**

Run: `python -m pytest tests/ml/test_incidents.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add ml/incidents.py tests/ml/test_incidents.py
git commit -m "feat(ml): registro de incidentes S0-S4 verificado con timestamp de apagón corregido a UTC"
```

---

### Task 2: Etiquetado graduado con pesos por severidad

**Files:**
- Modify: `ml/incidents.py`
- Test: `tests/ml/test_incidents.py`

**Interfaces:**
- Consume: `documented_incidents_utc()` (Task 1).
- Produce: `build_graded_labels(index: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]` → `(labels, weights)`
  donde `labels ∈ {0,1}` (1 = ventana previa a un evento mayor) y `weights` es el peso de muestra
  (3.0 dentro de ventana S4, 2.0 dentro de ventana S3, 1.0 negativos).

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/ml/test_incidents.py  (añadir)
import numpy as np


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
```

- [ ] **Step 2: Ejecutar el test para verificar que falla**

Run: `python -m pytest tests/ml/test_incidents.py::test_graded_labels_have_more_positives_than_old_single_window -v`
Expected: FAIL con `AttributeError: module 'ml.incidents' has no attribute 'build_graded_labels'`.

- [ ] **Step 3: Implementar `build_graded_labels`**

```python
# ml/incidents.py  (añadir)

def build_graded_labels(index: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """Etiqueta binaria de evento mayor con ventanas graduadas y pesos por severidad (D3).

    Returns:
        labels:  pd.Series[int] ∈ {0,1}; 1 en las `window_h` horas previas a cada evento.
        weights: pd.Series[float]; 3.0 en ventana de apagón (S4), 2.0 en ventana S3, 1.0 resto.
    """
    labels = pd.Series(0, index=index, dtype=int, name="event_label")
    weights = pd.Series(1.0, index=index, dtype=float, name="sample_weight")

    for inc in documented_incidents_utc():
        ts = inc["ts"]
        window_start = ts - pd.Timedelta(hours=inc["window_h"])
        mask = (index >= window_start) & (index < ts)
        labels[mask] = 1
        weights[mask] = 3.0 if inc["severity"] == S4_BLACKOUT else 2.0

    return labels, weights
```

- [ ] **Step 4: Ejecutar los tests para verificar que pasan**

Run: `python -m pytest tests/ml/test_incidents.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add ml/incidents.py tests/ml/test_incidents.py
git commit -m "feat(ml): etiquetado graduado de eventos con sample_weight por severidad"
```

---

### Task 3: Derivación de estados de estrés S0–S2 (función compartida con Plan 2)

**Files:**
- Modify: `ml/incidents.py`
- Test: `tests/ml/test_incidents.py`

**Interfaces:**
- Consume: `documented_incidents_utc()`, constantes de estado.
- Produce: `derive_stress_states(df: pd.DataFrame) -> pd.Series[int]` → estado S0–S4 por fila. Requiere las
  columnas `composite_pressure`, `renewables_pct`, `balance_mw` en `df` (presentes en
  `dataset_training.parquet`). **Esta función la consume el Plan 2 para la matriz de transición de Markov.**

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/ml/test_incidents.py  (añadir)

def _synthetic_df():
    idx = pd.date_range("2024-01-01", periods=1000, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "composite_pressure": rng.uniform(0, 1, size=1000),
        "renewables_pct": rng.uniform(0, 90, size=1000),
        "balance_mw": rng.normal(0, 4000, size=1000),
    }, index=idx)


def test_derive_states_returns_values_in_0_4():
    states = incidents.derive_stress_states(_synthetic_df())
    assert set(states.unique()).issubset({0, 1, 2, 3, 4})
    assert len(states) == 1000


def test_derive_states_s2_is_rarer_than_s1():
    states = incidents.derive_stress_states(_synthetic_df())
    n_s1 = (states == incidents.S1_STRESS).sum()
    n_s2 = (states == incidents.S2_NOTABLE).sum()
    # S2 usa un percentil más estricto (P97) que S1 (P90) → debe ser más raro
    assert 0 < n_s2 <= n_s1


def test_derive_states_overrides_with_documented_events():
    # Un df que cubre el apagón debe marcar S4 en su ventana
    idx = pd.date_range("2025-04-28 00:00", "2025-04-28 12:00", freq="h", tz="UTC")
    df = pd.DataFrame({
        "composite_pressure": 0.1, "renewables_pct": 30.0, "balance_mw": 0.0,
    }, index=idx)
    states = incidents.derive_stress_states(df)
    assert (states == incidents.S4_BLACKOUT).any()
```

- [ ] **Step 2: Ejecutar el test para verificar que falla**

Run: `python -m pytest tests/ml/test_incidents.py::test_derive_states_returns_values_in_0_4 -v`
Expected: FAIL con `AttributeError: ... has no attribute 'derive_stress_states'`.

- [ ] **Step 3: Implementar `derive_stress_states`**

```python
# ml/incidents.py  (añadir)

def derive_stress_states(df: pd.DataFrame) -> pd.Series:
    """Asigna estados S0–S4 por fila (shift-left, sin LLM; D2).

    - S0/S1/S2 se derivan por umbrales de percentil sobre proxies físicos del propio df.
    - S3/S4 se sobreescriben desde los eventos documentados (tienen prioridad sobre los umbrales).

    Requiere columnas: composite_pressure, renewables_pct, balance_mw.
    """
    states = pd.Series(S0_NORMAL, index=df.index, dtype=int, name="grid_state")

    cp = df["composite_pressure"]
    ren = df["renewables_pct"]
    bal = df["balance_mw"].abs()

    p90_cp, p97_cp = cp.quantile(0.90), cp.quantile(0.97)
    p90_ren = ren.quantile(0.90)
    p85_bal = bal.quantile(0.85)

    # S1 — estrés: alta renovable junto a desequilibrio, o presión compuesta alta
    s1_mask = ((ren > p90_ren) & (bal > p85_bal)) | (cp > p90_cp)
    states[s1_mask] = S1_STRESS

    # S2 — notable: presión compuesta severa (umbral más estricto que S1)
    s2_mask = cp > p97_cp
    states[s2_mask] = S2_NOTABLE

    # S3/S4 — eventos documentados: prioridad absoluta sobre los umbrales
    for inc in documented_incidents_utc():
        ts = inc["ts"]
        window_start = ts - pd.Timedelta(hours=inc["window_h"])
        mask = (df.index >= window_start) & (df.index <= ts)
        states[mask] = inc["severity"]

    return states
```

- [ ] **Step 4: Ejecutar los tests para verificar que pasan**

Run: `python -m pytest tests/ml/test_incidents.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add ml/incidents.py tests/ml/test_incidents.py
git commit -m "feat(ml): derive_stress_states S0-S4 por umbrales fisicos (compartido con Markov)"
```

---

### Task 4: Verificar/completar los jobs de entrenamiento de fondo

Operación, no TDD. El handoff lanzó dos jobs GPU; a fecha del plan solo existen `anomaly_detector.pkl` y
`blackout_classifier.pkl` en `ml/models/candidates/`. Faltan `lstm_ae.pth` y `load_forecaster.pkl`.

**Files:**
- Inspect: `ml/models/candidates/`

- [ ] **Step 1: Comprobar qué candidatos existen**

Run: `python -c "from pathlib import Path; print([p.name for p in Path('ml/models/candidates').glob('*')])"`
Expected: lista. Si **no** contiene `lstm_ae.pth`, ejecutar Step 2. Si **no** contiene `load_forecaster.pkl`,
ejecutar Step 3.

- [ ] **Step 2: (Si falta) re-lanzar el entrenamiento de anomalías (incluye LSTM) en GPU**

Run: `python -m ml.train.train_anomaly --if-trials 50 --lstm-trials 150`
Expected: al terminar, `ml/models/candidates/lstm_ae.pth` existe. Verás logs de Optuna y
`torch.cuda.is_available() → True` en la cabecera de `device_config`.

- [ ] **Step 3: (Si falta) re-lanzar el entrenamiento del forecaster en GPU**

Run: `python -m ml.train.train_forecast --trials 500`
Expected: al terminar, `ml/models/candidates/load_forecaster.pkl` existe.

- [ ] **Step 4: Confirmar los 4 candidatos**

Run: `python -c "from pathlib import Path; req={'anomaly_detector.pkl','lstm_ae.pth','load_forecaster.pkl','blackout_classifier.pkl'}; have={p.name for p in Path('ml/models/candidates').glob('*')}; print('OK' if req<=have else f'FALTAN: {req-have}')"`
Expected: `OK`.

No hay commit (no se modifica código; los `.pkl`/`.pth` no se versionan).

---

### Task 5: Sustituir walk-forward por Leave-One-Event-Out en `train_classifier.py`

**Files:**
- Modify: `ml/train/train_classifier.py`
- Test: `tests/ml/test_train_classifier.py`

**Interfaces:**
- Consume: `ml.incidents.build_graded_labels`, `build_features` (existente en `train_classifier.py:94`).
- Produce: `leave_one_event_out_splits(index: pd.DatetimeIndex, labels: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]`
  → lista de `(train_mask, val_mask)` booleanos, garantizando ≥1 positivo en cada `val_mask`.

- [ ] **Step 1: Escribir el test que falla**

```python
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
```

- [ ] **Step 2: Ejecutar el test para verificar que falla**

Run: `python -m pytest tests/ml/test_train_classifier.py::test_loeo_every_val_fold_has_a_positive -v`
Expected: FAIL con `AttributeError: module 'ml.train.train_classifier' has no attribute 'leave_one_event_out_splits'`.

- [ ] **Step 3: Implementar `leave_one_event_out_splits` y cablear el etiquetado nuevo**

En `ml/train/train_classifier.py`, eliminar `KNOWN_INCIDENTS` (líneas 49–54) y `build_labels`
(líneas 73–91). Añadir el import y la función:

```python
# ml/train/train_classifier.py  (cabecera, junto a los demás imports)
from ml.incidents import build_graded_labels


def leave_one_event_out_splits(
    index: pd.DatetimeIndex, labels: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Validación dejando un evento documentado fuera cada vez (D4).

    Cada fold valida sobre la ventana de un evento (± 12 h de margen) y entrena con el resto.
    Garantiza ≥1 positivo en validación, evitando el AUC=0 de walk-forward con eventos raros.
    """
    from ml.incidents import documented_incidents_utc

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for inc in documented_incidents_utc():
        ts = inc["ts"]
        val_start = ts - pd.Timedelta(hours=inc["window_h"] + 12)
        val_end = ts + pd.Timedelta(hours=12)
        val_mask = ((index >= val_start) & (index <= val_end)).values
        train_mask = ~val_mask
        if labels[val_mask].sum() > 0 and labels[train_mask].sum() > 0:
            splits.append((train_mask, val_mask))
    return splits
```

- [ ] **Step 4: Ejecutar el test para verificar que pasa**

Run: `python -m pytest tests/ml/test_train_classifier.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ml/train/train_classifier.py tests/ml/test_train_classifier.py
git commit -m "feat(ml): validacion Leave-One-Event-Out sustituye walk-forward para eventos raros"
```

---

### Task 6: Cablear etiquetado graduado, `sample_weight` y LOEO en el pipeline `run()`

**Files:**
- Modify: `ml/train/train_classifier.py` (función `run`, líneas 229–352; objetivo Optuna, líneas 130–182)
- Test: `tests/ml/test_train_classifier.py`

**Interfaces:**
- Produce: bundle joblib en `ml/models/candidates/blackout_classifier.pkl` con claves
  `{"model", "feature_cols", "best_params", "optimal_threshold", "scale_pos_weight", "metrics", "feature_importance"}`
  (mismo contrato que hoy, para no romper `validate_models.py` ni `retrain_pipeline.py`).

- [ ] **Step 1: Escribir el test que falla (forma del flujo de etiquetado en `run`)**

```python
# tests/ml/test_train_classifier.py  (añadir)

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
    assert captured.get("n_pos", 0) >= 12  # ≥ las 12 horas-evento graduadas
```

- [ ] **Step 2: Ejecutar el test para verificar que falla**

Run: `python -m pytest tests/ml/test_train_classifier.py::test_run_uses_graded_labels_and_weights -v`
Expected: FAIL (todavía usa `build_labels`/walk-forward; `n_pos` será 6 o el flujo lanzará error de fold vacío).

- [ ] **Step 3: Reescribir el cuerpo de etiquetado, splits y fit en `run()`**

Sustituir el bloque de etiquetado y splits (líneas ~249–264) por:

```python
    log.info("Construyendo features y etiquetas graduadas…")
    df_feat = build_features(df)
    labels, weights = build_graded_labels(df_feat.index)

    feature_cols = [c for c in df_feat.columns if c != "blackout_label"]
    X_all = df_feat[feature_cols].values.astype(np.float32)
    y_all = labels.reindex(df_feat.index).fillna(0).values.astype(int)
    w_all = weights.reindex(df_feat.index).fillna(1.0).values.astype(np.float32)

    n_neg = (y_all == 0).sum()
    n_pos = (y_all == 1).sum()
    scale_pos_weight = n_neg / max(n_pos, 1)
    log.info("Desbalance: %d neg / %d pos → scale_pos_weight base %.1f", n_neg, n_pos, scale_pos_weight)

    splits = leave_one_event_out_splits(df_feat.index, y_all)
    log.info("Leave-One-Event-Out: %d folds (uno por evento)", len(splits))
    if not splits:
        raise RuntimeError("Sin folds válidos — revisar ml/incidents.py y el rango del dataset")
```

En `make_objective` (línea 130), cambiar la firma para aceptar pesos y pasarlos al `fit`:

```python
def make_objective(X_full, y_full, w_full, splits, scale_pos_weight):
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score
    xgb_device = "cuda" if (DEVICE is not None and str(DEVICE) == "cuda") else "cpu"

    def objective(trial) -> float:
        params = { ... }  # idéntico al actual (líneas 138-155)
        fold_aucs = []
        for fold_idx, (train_mask, val_mask) in enumerate(splits):
            X_tr, y_tr, w_tr = X_full[train_mask], y_full[train_mask], w_full[train_mask]
            X_vl, y_vl = X_full[val_mask], y_full[val_mask]
            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_vl, y_vl)], verbose=False)
            proba = model.predict_proba(X_vl)[:, 1]
            fold_aucs.append(roc_auc_score(y_vl, proba))
            trial.report(np.mean(fold_aucs), fold_idx)
            if trial.should_prune():
                raise __import__("optuna").exceptions.TrialPruned()
        return -float(np.mean(fold_aucs)) if fold_aucs else 0.0
    return objective
```

Nota: los `train_mask`/`val_mask` de LOEO ya son booleanos alineados al índice de `df_feat`, así que el
indexado por máscara sustituye al `idx_positions` previo (eliminar las líneas 158–162 originales).

Actualizar la llamada en `run` (línea 274) y el `final_model.fit` (línea 302) para pasar `w_all`:

```python
    study.optimize(make_objective(X_all, y_all, w_all, splits, scale_pos_weight),
                   n_trials=n_trials, show_progress_bar=True)
    ...
    final_model.fit(X_train, y_train, sample_weight=w_all[train_idx], verbose=False)
```

- [ ] **Step 4: Ejecutar el test para verificar que pasa**

Run: `python -m pytest tests/ml/test_train_classifier.py -v`
Expected: PASS. (El test entrena con `n_trials=1`; tarda < 1 min en GPU.)

- [ ] **Step 5: Commit**

```bash
git add ml/train/train_classifier.py tests/ml/test_train_classifier.py
git commit -m "feat(ml): pipeline del clasificador con etiquetas graduadas, sample_weight y LOEO"
```

---

### Task 7: Re-entrenar el clasificador y validar contra el golden test

Operación. Produce el `.pkl` re-entrenado y comprueba que discrimina apagón vs normal.

**Files:**
- Run: `ml/train/train_classifier.py`, `ml/train/validate_models.py`

- [ ] **Step 1: Re-entrenar con tuning real en GPU**

Run: `python -m ml.train.train_classifier --trials 500`
Expected: logs de `leave_one_event_out_splits: 3 folds`; al final
`Modelo calibrado guardado: ml/models/candidates/blackout_classifier.pkl`. El AUC medio LOEO debe ser > 0
(ya no 0). Anota el valor: con 3 eventos sigue siendo una señal débil (ver D5 — no sobreinterpretar).

- [ ] **Step 2: Validar los 4 modelos con el golden dataset**

Run: `python -m ml.train.validate_models`
Expected: resumen con `classifier PASS` (P(blackout) máx ≥ 0,60 en snapshots críticos y < 0,40 en el
escenario normal, según `validate_models.py:286`). `iforest` y `lstm` también deberían pasar tras Task 4.

- [ ] **Step 3: Commit (solo si se ajustó algún umbral de validación)**

Si `validate_models.py` no necesitó cambios, no hay commit. Si ajustaste algún criterio, commitéalo:

```bash
git add ml/train/validate_models.py
git commit -m "test(ml): ajuste de criterio del golden test para el clasificador re-entrenado"
```

---

### Task 8: ~~Unificar el nombre del forecaster (D6)~~ — CANCELADA

**Cancelada el 2026-06-30 tras verificación en ejecución.** El handoff afirmaba una discrepancia de nombres
(`demand_forecaster.pkl` vs `load_forecaster.pkl`), pero `grep load_forecaster|demand_forecaster` sobre el
código muestra que `load_forecaster.pkl` es el nombre canónico y **ya es consistente** en todos los
consumidores:

- `ml/train/train_forecast.py:240` lo produce,
- `ml/retrain_pipeline.py:46` lo promueve (`PRODUCTION_MODELS`),
- `mcp_servers/ml_models_server.py:99` lo carga para inferencia,
- `ARCHITECTURE.md:544` lo documenta.

`demand_forecaster.pkl` no aparece en ningún punto del código — solo en el handoff. No hay promoción
silenciosamente saltada. Renombrar tocaría 4 sitios sin beneficio e introduciría riesgo (YAGNI, D6). **No se
realiza ningún cambio.** El nombre canónico es `load_forecaster.pkl`.

---

## Cierre del Plan 1

Al terminar: `ml/incidents.py` es la fuente de verdad de incidentes/estados; el clasificador se entrena con
etiquetas correctas (apagón a 10:33 UTC), 3 eventos, ventanas graduadas, `sample_weight` por severidad y
validación LOEO; `validate_models` pasa; los nombres de modelo son consistentes. **Prerrequisito cumplido
para el Plan 2** (que consume `derive_stress_states` y `documented_incidents_utc`).

Run final de regresión: `python -m pytest tests/ml/ -v`
