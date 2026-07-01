"""ml/data/incremental_load.py — Carga incremental de snapshots a la DB.

Puebla la tabla `snapshots` (SQLite de memory_server) con datos de ESIOS en la
MISMA escala que el dataset de entrenamiento, para que las features rolling de
serve tengan paridad con las de train (ver ml/features.py y ml/serve_features.py).

Principio de paridad: NO se reimplementa el parsing de ESIOS. Se reutiliza el ETL
canónico (ml/data/download_esios.py: suma de geo_ids, IDs verificados, x-api-key)
y las transformaciones de ml/data/build_dataset.py (resample horario, ÷2 de la
demanda). Así los snapshots quedan en la escala exacta del entrenamiento.

Uso:
    from ml.data.incremental_load import catch_up
    catch_up()                      # rellena huecos hasta ahora (bootstrap 30d si vacía)

    python -m ml.data.incremental_load --bootstrap-days 30
"""
from __future__ import annotations

import argparse
import json
import logging
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger("incremental_load")

BOOTSTRAP_DAYS = 30

_GEN_COLS = ["nuclear_mw", "hydro_mw", "wind_mw", "solar_pv_mw", "solar_thermal_mw",
             "gas_cc_mw", "coal_mw", "cogeneration_mw", "pumping_mw"]
_FLOW_COLS = ["es_fr_mw", "es_pt_mw", "es_ma_mw"]
_RESERVE_COLS = ["secondary_reserve_up_mwh", "secondary_reserve_dn_mwh"]


# ---------------------------------------------------------------------------
# Lógica pura (testeable en frío, sin ESIOS)
# ---------------------------------------------------------------------------

def _to_utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


def missing_hours(last_ts, now, bootstrap_days: int = BOOTSTRAP_DAYS) -> list[pd.Timestamp]:
    """Horas UTC faltantes en (last_ts, now]. Si last_ts is None → bootstrap."""
    now = _to_utc(now).floor("h")
    if last_ts is None:
        start = now - pd.Timedelta(days=bootstrap_days)
    else:
        start = _to_utc(last_ts).floor("h") + pd.Timedelta(hours=1)
    if start > now:
        return []
    return list(pd.date_range(start=start, end=now, freq="h", tz="UTC"))


def _num(row, col) -> float:
    if col not in row.index:
        return 0.0
    v = row[col]
    return float(v) if pd.notna(v) else 0.0


def raw_to_snapshots(df: pd.DataFrame) -> list[dict]:
    """Convierte un DataFrame horario en escala de entrenamiento (salida de
    build_dataset.load_esios) en snapshots con el esquema de la tabla `snapshots`.
    """
    snaps = []
    for ts, row in df.iterrows():
        reserve = abs(_num(row, "secondary_reserve_up_mwh")) + abs(_num(row, "secondary_reserve_dn_mwh"))
        snaps.append({
            "timestamp": _to_utc(ts).isoformat(),
            "region": "peninsular",
            "demand_mw": _num(row, "demand_real_mw"),
            "generation_by_tech": {c: _num(row, c) for c in _GEN_COLS if c in df.columns},
            "interchange_flows": {c: _num(row, c) for c in _FLOW_COLS if c in df.columns},
            "reserve_mw": float(reserve),
        })
    return snaps


# ---------------------------------------------------------------------------
# Persistencia (dedup por timestamp)
# ---------------------------------------------------------------------------

def _store_snapshots(snaps: list[dict]) -> int:
    from mcp_servers import memory_server
    with memory_server._conn() as con:
        existing = {r[0] for r in con.execute("SELECT timestamp FROM snapshots").fetchall()}
        n = 0
        for s in snaps:
            if s["timestamp"] in existing:
                continue
            con.execute(
                """INSERT INTO snapshots
                   (timestamp, region, demand_mw, generation_json, interchange_json, reserve_mw)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (s["timestamp"], s.get("region", "peninsular"), s["demand_mw"],
                 json.dumps(s["generation_by_tech"]), json.dumps(s["interchange_flows"]),
                 s["reserve_mw"]),
            )
            n += 1
    return n


def _last_snapshot_ts():
    from mcp_servers import memory_server
    with memory_server._conn() as con:
        row = con.execute("SELECT MAX(timestamp) AS mx FROM snapshots").fetchone()
    return row["mx"] if row and row["mx"] else None


# ---------------------------------------------------------------------------
# ESIOS live (reutiliza el ETL canónico)
# ---------------------------------------------------------------------------

def backfill(start_date: date, end_date: date, raw_dir: Path | None = None) -> int:
    """Descarga [start_date, end_date] vía download_esios, aplica las transformaciones
    de build_dataset (escala de entrenamiento) y persiste los snapshots. Devuelve nº
    de snapshots nuevos insertados."""
    from ml.data import download_esios, build_dataset

    tmp = Path(raw_dir) if raw_dir else Path(tempfile.mkdtemp(prefix="esios_backfill_"))
    log.info("Backfill ESIOS %s → %s (raw en %s)", start_date, end_date, tmp)
    download_esios.run(start_date, end_date, out_dir=tmp)

    df = build_dataset.load_esios(tmp)  # resample horario + ÷2 demanda (escala train)
    if df.empty:
        log.warning("load_esios devolvió DataFrame vacío para %s → %s", start_date, end_date)
        return 0

    snaps = raw_to_snapshots(df)
    inserted = _store_snapshots(snaps)
    log.info("Backfill: %d snapshots nuevos (de %d horas)", inserted, len(snaps))
    return inserted


def catch_up(now=None, bootstrap_days: int = BOOTSTRAP_DAYS) -> int:
    """Rellena la tabla snapshots desde el último timestamp almacenado hasta `now`.
    Si la tabla está vacía, hace bootstrap de `bootstrap_days` días."""
    now = now or datetime.now(timezone.utc)
    hours = missing_hours(_last_snapshot_ts(), now, bootstrap_days)
    if not hours:
        log.info("catch_up: sin huecos que rellenar")
        return 0
    return backfill(hours[0].date(), hours[-1].date())


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Carga incremental de snapshots ESIOS")
    p.add_argument("--bootstrap-days", type=int, default=BOOTSTRAP_DAYS)
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    args = _parse_args()
    n = catch_up(bootstrap_days=args.bootstrap_days)
    log.info("Carga incremental completada: %d snapshots nuevos", n)
