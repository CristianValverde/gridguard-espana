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
