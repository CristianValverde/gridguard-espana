# Plan 2 — Capa de cadena de Markov (Python puro)

> **Para trabajadores agénticos:** SUB-SKILL REQUERIDO: `superpowers:subagent-driven-development` o
> `superpowers:executing-plans`. Lee el [índice maestro](2026-06-30-gridguard-incident-markov-agents-INDEX.md)
> (constraints, D1–D6) y completa el [Plan 1](2026-06-30-plan1-incident-enrichment-classifier.md) antes
> (este plan consume `ml.incidents.derive_stress_states` y `documented_incidents_utc`).

**Objetivo:** añadir una capa de post-procesado estadística (sin LLM) que convierte el estado puntual de la
red en una **trayectoria de riesgo**: dado el estado actual S0–S4 y la matriz de transición empírica, estimar
hacia dónde evoluciona el sistema en las próximas N horas. No reemplaza al clasificador; le añade memoria
entre instantes.

**Arquitectura:** dos módulos con responsabilidades separadas, siguiendo la convención del repo (`ml/` entrena
offline, `tools/` infiere en runtime):
- `ml/grid_state_tracker.py` — **offline**: estima la matriz de transición A[5×5] de los estados derivados,
  la mezcla con priors expertos (Dirichlet) para transiciones raras de alta severidad, y persiste matriz +
  umbrales a `ml/models/markov_state.npz`.
- `tools/markov_state_tracker.py` — **runtime, sin LLM**: carga el `.npz`, discretiza el estado actual
  (`compute_state`), devuelve la matriz (`get_transition_matrix`) y proyecta la trayectoria (`get_trajectory`).
  Es el tool que el `markov_server` MCP (Plan 3) envolverá.

**Tech stack:** NumPy, pandas, pytest. Sin GPU, sin LLM.

## Constraints globales

Ver [índice maestro](2026-06-30-gridguard-incident-markov-agents-INDEX.md). Clave para este plan: shift-left
estricto (todo Python/NumPy determinista); UTC; los umbrales de runtime DEBEN ser los mismos que en la
estimación offline (se persisten juntos para evitar deriva entrenamiento/inferencia).

---

## Estructura de ficheros

- **Crear** `ml/grid_state_tracker.py` — estimación + persistencia de la matriz de Markov.
- **Crear** `tests/ml/test_grid_state_tracker.py`.
- **Crear** `tools/markov_state_tracker.py` — inferencia de estado y trayectoria en runtime.
- **Crear** `tests/tools/test_markov_state_tracker.py`.
- **Artefacto generado** `ml/models/markov_state.npz` — matriz A[5×5] + umbrales (no versionado).

---

### Task 1: Estimación de la matriz de transición empírica con suavizado de Laplace

**Files:**
- Create: `ml/grid_state_tracker.py`
- Test: `tests/ml/test_grid_state_tracker.py`

**Interfaces:**
- Consume: `ml.incidents` (constantes de estado).
- Produce: `estimate_transition_matrix(states: pd.Series, n_states: int = 5, alpha: float = 1.0) -> np.ndarray`
  → matriz `A[n_states, n_states]` fila-estocástica (cada fila suma 1), `A[i, j] = P(S_{t+1}=j | S_t=i)`.

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/ml/test_grid_state_tracker.py
import numpy as np
import pandas as pd
from ml import grid_state_tracker as gst


def test_transition_matrix_is_row_stochastic():
    states = pd.Series([0, 0, 1, 1, 0, 1, 2, 1, 0, 0])
    A = gst.estimate_transition_matrix(states, n_states=5, alpha=1.0)
    assert A.shape == (5, 5)
    np.testing.assert_allclose(A.sum(axis=1), np.ones(5), rtol=1e-9)
    assert (A >= 0).all()


def test_transition_matrix_reflects_observed_counts():
    # Secuencia 0->1 cuatro veces, nunca 0->4. Con Laplace alpha pequeño,
    # P(1|0) debe ser >> P(4|0).
    states = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    A = gst.estimate_transition_matrix(states, n_states=5, alpha=0.01)
    assert A[0, 1] > A[0, 4]
```

- [ ] **Step 2: Ejecutar el test para verificar que falla**

Run: `python -m pytest tests/ml/test_grid_state_tracker.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'ml.grid_state_tracker'`.

- [ ] **Step 3: Implementar `estimate_transition_matrix`**

```python
"""
ml/grid_state_tracker.py — Estimación offline de la cadena de Markov de estados de red.

Shift-left: pura estadística, sin LLM. Estima A[5x5] de los estados S0–S4 derivados por
ml.incidents.derive_stress_states, la mezcla con priors expertos para transiciones raras de
alta severidad, y persiste matriz + umbrales para uso en runtime (tools/markov_state_tracker.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.incidents import (
    S0_NORMAL, S1_STRESS, S2_NOTABLE, S3_MAJOR, S4_BLACKOUT,
)

N_STATES = 5
MODEL_PATH = Path("ml/models/markov_state.npz")


def estimate_transition_matrix(
    states: pd.Series, n_states: int = N_STATES, alpha: float = 1.0,
) -> np.ndarray:
    """Matriz de transición empírica con suavizado de Laplace.

    A[i, j] = (conteo(i->j) + alpha) / sum_j(conteo(i->j) + alpha)
    """
    counts = np.full((n_states, n_states), alpha, dtype=np.float64)
    s = states.values.astype(int)
    for a, b in zip(s[:-1], s[1:]):
        counts[a, b] += 1.0
    return counts / counts.sum(axis=1, keepdims=True)
```

- [ ] **Step 4: Ejecutar el test para verificar que pasa**

Run: `python -m pytest tests/ml/test_grid_state_tracker.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add ml/grid_state_tracker.py tests/ml/test_grid_state_tracker.py
git commit -m "feat(ml): matriz de transicion de Markov con suavizado de Laplace"
```

---

### Task 2: Priors expertos (Dirichlet) para transiciones raras de alta severidad

**Files:**
- Modify: `ml/grid_state_tracker.py`
- Test: `tests/ml/test_grid_state_tracker.py`

**Interfaces:**
- Produce: `PRIOR_MATRIX: np.ndarray[5,5]` (matriz de probabilidad prior experta);
  `blend_with_prior(empirical_counts: np.ndarray, kappa: float = 5.0) -> np.ndarray` → matriz fila-estocástica
  que mezcla conteos empíricos con el prior con fuerza `kappa` (pseudo-observaciones).

**Justificación del prior (D5, mitigación de escasez):** S2→S3 y S3→S4 tienen ~0–1 observaciones en 6 años.
Sin prior, Laplace las haría uniformes (irreal: implicaría P(apagón|incidente mayor) ≈ 20 %). El prior experto
codifica que estas transiciones son raras pero no uniformes, usando órdenes de magnitud de la estadística de
disturbios ENTSO-E (la mayoría de incidentes mayores NO escalan a apagón). Valores conservadores y explícitos.

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/ml/test_grid_state_tracker.py  (añadir)

def test_prior_matrix_is_row_stochastic():
    np.testing.assert_allclose(gst.PRIOR_MATRIX.sum(axis=1), np.ones(5), rtol=1e-9)


def test_blend_pulls_rare_transition_toward_prior():
    # Sin observaciones de 3->4, la probabilidad mezclada debe acercarse al prior,
    # no a 1/5 (uniforme de Laplace).
    counts = np.zeros((5, 5))
    counts[0, 0] = 1000  # mucha masa en S0->S0
    A = gst.blend_with_prior(counts, kappa=5.0)
    np.testing.assert_allclose(A.sum(axis=1), np.ones(5), rtol=1e-9)
    # P(S4|S3) mezclado ≈ prior S3->S4, claramente < uniforme 0.2
    assert A[gst.S3_MAJOR, gst.S4_BLACKOUT] < 0.2
    assert A[gst.S3_MAJOR, gst.S4_BLACKOUT] == gst.PRIOR_MATRIX[gst.S3_MAJOR, gst.S4_BLACKOUT]
```

- [ ] **Step 2: Ejecutar el test para verificar que falla**

Run: `python -m pytest tests/ml/test_grid_state_tracker.py::test_blend_pulls_rare_transition_toward_prior -v`
Expected: FAIL (`PRIOR_MATRIX`/`blend_with_prior` no existen).

- [ ] **Step 3: Implementar el prior y la mezcla**

```python
# ml/grid_state_tracker.py  (añadir)

# Prior experto fila-estocástico. Codifica conocimiento de dominio sobre la dinámica
# de la red: el sistema tiende a permanecer/relajarse; escalar a mayor severidad es raro.
# Filas: estado origen S0..S4. Columnas: estado destino S0..S4.
PRIOR_MATRIX = np.array([
    [0.90, 0.09, 0.009, 0.0009, 0.0001],  # S0 → casi siempre normal
    [0.40, 0.50, 0.090, 0.0090, 0.0010],  # S1 → relaja o persiste
    [0.10, 0.40, 0.450, 0.0450, 0.0050],  # S2 → mayormente baja/persiste
    [0.05, 0.15, 0.350, 0.4000, 0.0500],  # S3 → P(escalar a apagón) ≈ 5%
    [0.30, 0.30, 0.200, 0.1500, 0.0500],  # S4 → recuperación gradual
], dtype=np.float64)


def blend_with_prior(empirical_counts: np.ndarray, kappa: float = 5.0) -> np.ndarray:
    """Mezcla bayesiana (Dirichlet): conteos empíricos + kappa·prior como pseudo-conteos.

    Donde hay muchos datos, dominan los conteos; donde son escasos (S2→S3→S4), domina el prior.
    """
    pseudo = kappa * PRIOR_MATRIX
    blended = empirical_counts + pseudo
    return blended / blended.sum(axis=1, keepdims=True)
```

- [ ] **Step 4: Ejecutar el test para verificar que pasa**

Run: `python -m pytest tests/ml/test_grid_state_tracker.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add ml/grid_state_tracker.py tests/ml/test_grid_state_tracker.py
git commit -m "feat(ml): priors expertos Dirichlet para transiciones raras S2-S3-S4"
```

---

### Task 3: Ajuste end-to-end y persistencia de matriz + umbrales

**Files:**
- Modify: `ml/grid_state_tracker.py`
- Test: `tests/ml/test_grid_state_tracker.py`

**Interfaces:**
- Consume: `ml.incidents.derive_stress_states` (Plan 1), `dataset_training.parquet`.
- Produce: `fit_and_save(df: pd.DataFrame, out_path: Path = MODEL_PATH) -> dict` que computa estados, estima la
  matriz mezclada, **persiste** a `.npz` `{transition_matrix, thresholds, state_counts}` y devuelve un resumen.
  `thresholds` es un dict serializable con `p90_cp, p97_cp, p90_ren, p85_bal` (los mismos umbrales de
  `derive_stress_states`, para reutilizarlos en runtime sin el dataset).

**Nota de implementación:** para persistir los umbrales junto a la matriz, refactoriza `derive_stress_states`
del Plan 1 de modo que también exponga los umbrales. Mínimamente invasivo: añade en `ml/incidents.py` una
función `compute_thresholds(df) -> dict` y haz que `derive_stress_states` la use. Esto NO cambia el contrato
del Plan 1 (sus tests siguen pasando) y da a runtime los umbrales exactos.

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/ml/test_grid_state_tracker.py  (añadir)

def test_fit_and_save_roundtrip(tmp_path):
    idx = pd.date_range("2024-01-01", periods=2000, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "composite_pressure": rng.uniform(0, 1, 2000),
        "renewables_pct": rng.uniform(0, 90, 2000),
        "balance_mw": rng.normal(0, 4000, 2000),
    }, index=idx)

    out = tmp_path / "markov_state.npz"
    summary = gst.fit_and_save(df, out_path=out)
    assert out.exists()
    assert summary["transition_matrix"].shape == (5, 5)

    data = np.load(out, allow_pickle=True)
    A = data["transition_matrix"]
    np.testing.assert_allclose(A.sum(axis=1), np.ones(5), rtol=1e-9)
    thr = data["thresholds"].item()  # dict serializado
    assert {"p90_cp", "p97_cp", "p90_ren", "p85_bal"} <= set(thr)
```

- [ ] **Step 2: Ejecutar el test para verificar que falla**

Run: `python -m pytest tests/ml/test_grid_state_tracker.py::test_fit_and_save_roundtrip -v`
Expected: FAIL (`fit_and_save` no existe).

- [ ] **Step 3: Implementar `compute_thresholds` (en incidents) y `fit_and_save`**

En `ml/incidents.py` añade y refactoriza:

```python
# ml/incidents.py  (añadir)

def compute_thresholds(df: pd.DataFrame) -> dict:
    """Umbrales de percentil usados para derivar S0–S2. Persistidos para runtime."""
    return {
        "p90_cp": float(df["composite_pressure"].quantile(0.90)),
        "p97_cp": float(df["composite_pressure"].quantile(0.97)),
        "p90_ren": float(df["renewables_pct"].quantile(0.90)),
        "p85_bal": float(df["balance_mw"].abs().quantile(0.85)),
    }
```

Y haz que `derive_stress_states` (Task 3 del Plan 1) obtenga sus umbrales de `compute_thresholds(df)` en lugar
de calcularlos inline (mismo resultado; ahora reutilizable).

En `ml/grid_state_tracker.py`:

```python
# ml/grid_state_tracker.py  (añadir)
from ml.incidents import compute_thresholds, derive_stress_states


def fit_and_save(df: pd.DataFrame, out_path: Path = MODEL_PATH) -> dict:
    """Estima la matriz de Markov del dataset y la persiste con los umbrales de discretización."""
    states = derive_stress_states(df)
    counts = np.full((N_STATES, N_STATES), 0.0)
    s = states.values.astype(int)
    for a, b in zip(s[:-1], s[1:]):
        counts[a, b] += 1.0

    A = blend_with_prior(counts, kappa=5.0)
    thresholds = compute_thresholds(df)
    state_counts = np.bincount(s, minlength=N_STATES)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        transition_matrix=A,
        thresholds=np.array(thresholds, dtype=object),
        state_counts=state_counts,
    )
    return {"transition_matrix": A, "thresholds": thresholds, "state_counts": state_counts}


def _main() -> None:
    data = pd.read_parquet("data/processed/dataset_training.parquet")
    data.index = pd.DatetimeIndex(data.index, tz="UTC")
    summary = fit_and_save(data)
    print("Matriz de transición A[5x5]:")
    print(np.round(summary["transition_matrix"], 4))
    print("Distribución de estados:", summary["state_counts"])


if __name__ == "__main__":
    _main()
```

- [ ] **Step 4: Ejecutar los tests + regresión del Plan 1**

Run: `python -m pytest tests/ml/test_grid_state_tracker.py tests/ml/test_incidents.py -v`
Expected: PASS (todos — incluidos los del Plan 1, que no deben romperse por el refactor de `compute_thresholds`).

- [ ] **Step 5: Generar el artefacto real y commit**

Run: `python -m ml.grid_state_tracker`
Expected: imprime la matriz A[5×5] y la distribución de estados; crea `ml/models/markov_state.npz`.

```bash
git add ml/grid_state_tracker.py ml/incidents.py tests/ml/test_grid_state_tracker.py
git commit -m "feat(ml): fit_and_save de la cadena de Markov con umbrales persistidos"
```

---

### Task 4: Tool de runtime `tools/markov_state_tracker.py`

**Files:**
- Create: `tools/markov_state_tracker.py`
- Test: `tests/tools/test_markov_state_tracker.py`

**Interfaces:**
- Consume: `ml/models/markov_state.npz` (Task 3).
- Produce (firmas exactas que el `markov_server` del Plan 3 envolverá):
  - `compute_state(features: dict) -> int` — discretiza el estado actual S0–S4 desde
    `{composite_pressure, renewables_pct, balance_mw}` usando los umbrales persistidos.
  - `get_transition_matrix() -> np.ndarray` — devuelve A[5×5].
  - `get_trajectory(current_state: int, horizon_hours: int) -> list[int]` — secuencia del estado más probable
    para las próximas `horizon_hours` (proyección `v0 @ A^t`, argmax por paso).
  - `prob_reach_severity(current_state: int, target: int, horizon_hours: int) -> float` — P(alcanzar estado
    ≥ `target` en algún momento dentro del horizonte). Esto es lo que hace accionable la trayectoria
    ("en S1 durante 3 h → P(llegar a S3 en 6 h) = 18 %").

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/tools/test_markov_state_tracker.py
import numpy as np
import pytest
from tools import markov_state_tracker as mst


def test_compute_state_low_pressure_is_normal(tmp_path, monkeypatch):
    # Forzar un .npz de prueba con umbrales conocidos
    from ml import grid_state_tracker as gst
    thr = {"p90_cp": 0.8, "p97_cp": 0.95, "p90_ren": 80.0, "p85_bal": 6000.0}
    A = gst.blend_with_prior(np.eye(5) * 100)
    out = tmp_path / "markov_state.npz"
    np.savez(out, transition_matrix=A, thresholds=np.array(thr, dtype=object),
             state_counts=np.ones(5))
    monkeypatch.setattr(mst, "MODEL_PATH", out)
    mst._reset_cache()

    s = mst.compute_state({"composite_pressure": 0.1, "renewables_pct": 30, "balance_mw": 100})
    assert s == 0


def test_get_trajectory_length_and_range(tmp_path, monkeypatch):
    from ml import grid_state_tracker as gst
    thr = {"p90_cp": 0.8, "p97_cp": 0.95, "p90_ren": 80.0, "p85_bal": 6000.0}
    A = gst.blend_with_prior(np.eye(5) * 100)
    out = tmp_path / "markov_state.npz"
    np.savez(out, transition_matrix=A, thresholds=np.array(thr, dtype=object),
             state_counts=np.ones(5))
    monkeypatch.setattr(mst, "MODEL_PATH", out)
    mst._reset_cache()

    traj = mst.get_trajectory(current_state=1, horizon_hours=6)
    assert len(traj) == 6
    assert all(0 <= s <= 4 for s in traj)


def test_prob_reach_severity_is_between_0_and_1(tmp_path, monkeypatch):
    from ml import grid_state_tracker as gst
    out = tmp_path / "markov_state.npz"
    A = gst.blend_with_prior(np.eye(5) * 100)
    np.savez(out, transition_matrix=A,
             thresholds=np.array({"p90_cp": .8, "p97_cp": .95, "p90_ren": 80., "p85_bal": 6000.}, dtype=object),
             state_counts=np.ones(5))
    monkeypatch.setattr(mst, "MODEL_PATH", out)
    mst._reset_cache()

    p = mst.prob_reach_severity(current_state=2, target=3, horizon_hours=6)
    assert 0.0 <= p <= 1.0
```

- [ ] **Step 2: Ejecutar el test para verificar que falla**

Run: `python -m pytest tests/tools/test_markov_state_tracker.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'tools.markov_state_tracker'`.

- [ ] **Step 3: Implementar el tool de runtime**

```python
"""
tools/markov_state_tracker.py — Inferencia de estado y trayectoria de Markov en runtime.

Shift-left puro (sin LLM). Carga ml/models/markov_state.npz (matriz + umbrales) y expone:
discretización del estado actual, matriz de transición, y proyección de trayectoria.
Envuelto por mcp_servers/markov_server.py (Plan 3).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ml.incidents import S0_NORMAL, S1_STRESS, S2_NOTABLE, S3_MAJOR, S4_BLACKOUT

N_STATES = 5
MODEL_PATH = Path("ml/models/markov_state.npz")
_CACHE: dict | None = None


def _reset_cache() -> None:
    global _CACHE
    _CACHE = None


def _load() -> dict:
    global _CACHE
    if _CACHE is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"{MODEL_PATH} no existe. Ejecutar: python -m ml.grid_state_tracker"
            )
        data = np.load(MODEL_PATH, allow_pickle=True)
        _CACHE = {
            "A": data["transition_matrix"],
            "thr": data["thresholds"].item(),
        }
    return _CACHE


def compute_state(features: dict) -> int:
    """Discretiza el estado actual S0–S4 desde proxies, con los umbrales persistidos.

    Misma lógica que ml.incidents.derive_stress_states pero para un único punto (sin eventos).
    """
    thr = _load()["thr"]
    cp = float(features.get("composite_pressure", 0.0))
    ren = float(features.get("renewables_pct", 0.0))
    bal = abs(float(features.get("balance_mw", 0.0)))

    if cp > thr["p97_cp"]:
        return S2_NOTABLE
    if (ren > thr["p90_ren"] and bal > thr["p85_bal"]) or cp > thr["p90_cp"]:
        return S1_STRESS
    return S0_NORMAL


def get_transition_matrix() -> np.ndarray:
    return _load()["A"]


def get_trajectory(current_state: int, horizon_hours: int) -> list[int]:
    """Estado más probable en cada una de las próximas horizon_hours (proyección v0·A^t)."""
    A = _load()["A"]
    v = np.zeros(A.shape[0])
    v[int(current_state)] = 1.0
    traj = []
    for _ in range(int(horizon_hours)):
        v = v @ A
        traj.append(int(np.argmax(v)))
    return traj


def prob_reach_severity(current_state: int, target: int, horizon_hours: int) -> float:
    """P(estado ≥ target en algún paso dentro del horizonte). Útil para alertas accionables."""
    A = _load()["A"]
    v = np.zeros(A.shape[0])
    v[int(current_state)] = 1.0
    not_reached = 1.0
    for _ in range(int(horizon_hours)):
        v = v @ A
        p_step = float(v[int(target):].sum())
        not_reached *= (1.0 - p_step)
    return 1.0 - not_reached
```

- [ ] **Step 4: Ejecutar los tests para verificar que pasan**

Run: `python -m pytest tests/tools/test_markov_state_tracker.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add tools/markov_state_tracker.py tests/tools/test_markov_state_tracker.py
git commit -m "feat(tools): inferencia de estado y trayectoria de Markov en runtime"
```

---

## Cierre del Plan 2

Al terminar: existe `ml/models/markov_state.npz` con la matriz A[5×5] mezclada con priors; `tools/markov_state_tracker.py`
discretiza estado, devuelve la matriz y proyecta trayectorias + probabilidad de escalada. **Prerrequisito
cumplido para el Plan 3** (`markov_server` envuelve estas 4 funciones) y el Plan 4 (los agentes consumen
`get_grid_state`/`get_markov_trajectory`).

Run de regresión: `python -m pytest tests/ml/ tests/tools/ -v`
