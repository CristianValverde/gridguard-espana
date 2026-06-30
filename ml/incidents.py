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
