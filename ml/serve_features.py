"""Ensamblado serve-time del feature store con estado.

Reconstruye la ventana trailing (168h) desde snapshots crudos (DB o golden),
la lleva a índice horario UTC y aplica ml.features.build_features. Fuente-
agnóstico: acepta cualquier lista de snapshots dicts con el esquema de la tabla
`snapshots` (demand_mw, generation_by_tech, interchange_flows, reserve_mw).

La paridad train/serve la garantiza el test tests/ml/test_serve_features.py:
la fila de serve debe coincidir con la fila de build_features sobre el dataset
completo en el mismo timestamp.
"""
from __future__ import annotations

import json

import pandas as pd

from ml import features as F

_LINKS = ("es_fr_mw", "es_pt_mw", "es_ma_mw")


def db_rows_to_snapshots(rows) -> list[dict]:
    """Convierte filas de la tabla `snapshots` (dicts o sqlite.Row con las columnas
    generation_json / interchange_json) en snapshots dict para serve_feature_row.
    Es el puente DB→serve; mantiene serve_features agnóstico del origen."""
    out = []
    for r in rows:
        out.append({
            "timestamp": r["timestamp"],
            "demand_mw": r["demand_mw"],
            "generation_by_tech": json.loads(r["generation_json"] or "{}"),
            "interchange_flows": json.loads(r["interchange_json"] or "{}"),
            "reserve_mw": r["reserve_mw"],
        })
    return out


def assemble_window(snapshots: list[dict], now: pd.Timestamp) -> pd.DataFrame:
    """Reconstruye la ventana horaria [now-168h, now] a partir de snapshots crudos."""
    if not snapshots:
        raise ValueError("assemble_window requiere al menos un snapshot")

    rows = []
    for s in snapshots:
        gen = s.get("generation_by_tech", {}) or {}
        flows = s.get("interchange_flows", {}) or {}
        row = {
            "timestamp": pd.Timestamp(s["timestamp"]),
            "demand_real_mw": float(s.get("demand_mw") or 0.0),
            "reserve_mw": float(s.get("reserve_mw") or 0.0),
        }
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

    now = pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tz is None else now.tz_convert("UTC")
    start = now - pd.Timedelta(hours=F.ROLLING_WINDOW_H)
    full_idx = pd.date_range(start=start, end=now, freq="h", tz="UTC")

    # Índice horario continuo hasta `now`; huecos pequeños → ffill.
    df = df.reindex(df.index.union(full_idx)).sort_index()
    df = df.reindex(full_idx, method="ffill")
    df.index.name = "timestamp"
    return df


def serve_feature_frame(snapshots: list[dict], now: pd.Timestamp) -> pd.DataFrame:
    """assemble_window → build_features → DataFrame completo con FEATURE_COLS.

    Para modelos de secuencia/ventana (LSTM AE, Isolation Forest): misma
    canalización que serve_feature_row pero devolviendo todas las filas."""
    df_raw = assemble_window(snapshots, now)
    return F.build_features(df_raw)[F.FEATURE_COLS]


def serve_feature_row(snapshots: list[dict], now: pd.Timestamp) -> dict:
    """assemble_window → build_features → última fila → dict con FEATURE_COLS."""
    last = serve_feature_frame(snapshots, now).iloc[-1]
    return {c: float(last[c]) for c in F.FEATURE_COLS}
