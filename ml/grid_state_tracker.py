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
    compute_thresholds, derive_stress_states,
)

N_STATES = 5
MODEL_PATH = Path("ml/models/markov_state.npz")

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
PRIOR_MATRIX.flags.writeable = False


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


def blend_with_prior(empirical_counts: np.ndarray, kappa: float = 5.0) -> np.ndarray:
    """Mezcla bayesiana (Dirichlet): conteos empíricos + kappa·prior como pseudo-conteos.

    Donde hay muchos datos, dominan los conteos; donde son escasos (S2→S3→S4), domina el prior.
    """
    if kappa <= 0:
        raise ValueError(f"kappa must be > 0, got {kappa}")
    pseudo = kappa * PRIOR_MATRIX
    blended = empirical_counts + pseudo
    return blended / blended.sum(axis=1, keepdims=True)


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

    out_path = Path(out_path)
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
