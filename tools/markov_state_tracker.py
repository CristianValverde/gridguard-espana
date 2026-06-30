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
