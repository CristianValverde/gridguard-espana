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
