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
