"""
mcp_servers/memory_server.py — MCP: memory

Persistencia local con SQLite. Almacena snapshots, risk scores e incidentes.
Ruta de la BD: variable de entorno DB_PATH (default: ./data/gridguard.db).

Herramientas expuestas:
  get_baseline(window_hours)                       -> Baseline (JSON)
  query_incidents(start, end)                      -> list[Incident] (JSON)
  get_risk_history(hours)                          -> list[RiskScore] (JSON)

La herramienta store_risk_score se registra también
para que otros agentes puedan persistir sus outputs.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator

from mcp.server.fastmcp import FastMCP

from ml.incidents import STATE_LABELS
from schemas.models import Baseline, Incident, Lesson, RiskScore

mcp = FastMCP("gridguard-memory")

_DEFAULT_DB = "./data/gridguard.db"


def _db_path() -> Path:
    p = Path(os.environ.get("DB_PATH", _DEFAULT_DB))
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


@contextmanager
def _conn() -> Generator[sqlite3.Connection, None, None]:
    con = sqlite3.connect(str(_db_path()), detect_types=sqlite3.PARSE_DECLTYPES)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def _init_db() -> None:
    """Crea tablas si no existen. Llamado al arrancar el servidor."""
    with _conn() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS snapshots (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp        TEXT    NOT NULL,
                region           TEXT    NOT NULL DEFAULT 'peninsular',
                demand_mw        REAL,
                generation_json  TEXT,
                interchange_json TEXT,
                reserve_mw       REAL,
                created_at       TEXT    DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_snapshots_ts ON snapshots(timestamp);

            CREATE TABLE IF NOT EXISTS risk_scores (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp            TEXT    NOT NULL,
                value                INTEGER NOT NULL,
                level                TEXT    NOT NULL,
                blackout_probability REAL,
                components_json      TEXT,
                created_at           TEXT    DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_risk_ts ON risk_scores(timestamp);

            CREATE TABLE IF NOT EXISTS incidents (
                incident_id          TEXT    PRIMARY KEY,
                start                TEXT    NOT NULL,
                end                  TEXT,
                severity             TEXT    NOT NULL,
                description          TEXT,
                resolution           TEXT,
                risk_score_at_onset  INTEGER DEFAULT 0,
                created_at           TEXT    DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS grid_states (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp  TEXT    NOT NULL,
                state      INTEGER NOT NULL,
                created_at TEXT    DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_grid_states_ts ON grid_states(timestamp);

            CREATE TABLE IF NOT EXISTS lessons (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp     TEXT    NOT NULL,
                markov_state  INTEGER NOT NULL,
                p_band        TEXT    NOT NULL,
                incident_type TEXT,
                incident_id   TEXT,
                source_agent  TEXT    NOT NULL,
                context       TEXT    NOT NULL,
                lesson        TEXT    NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_lessons_ts ON lessons(timestamp);
        """)


# ---------------------------------------------------------------------------
# Herramientas MCP
# ---------------------------------------------------------------------------

@mcp.tool()
def store_risk_score(risk_score_json: str) -> dict:
    """Persiste un RiskScore en la BD local.

    Args:
        risk_score_json: RiskScore serializado como JSON string
    """
    try:
        data = json.loads(risk_score_json)
        with _conn() as con:
            con.execute(
                """INSERT INTO risk_scores
                   (timestamp, value, level, blackout_probability, components_json)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    data.get("timestamp"),
                    data.get("value"),
                    data.get("level"),
                    data.get("blackout_probability"),
                    json.dumps(data.get("components", {})),
                ),
            )
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@mcp.tool()
def get_baseline(window_hours: int = 168) -> dict:
    """Calcula estadísticas de referencia histórica (media/std) para el período dado.

    Args:
        window_hours: Horas hacia atrás para calcular el baseline (default 168 = 7 días)
    """
    try:
        cutoff = (
            datetime.now(timezone.utc) - timedelta(hours=window_hours)
        ).isoformat()

        with _conn() as con:
            rows = con.execute(
                "SELECT demand_mw, generation_json, interchange_json, reserve_mw "
                "FROM snapshots WHERE timestamp >= ? ORDER BY timestamp",
                (cutoff,),
            ).fetchall()

        if not rows:
            return {"error": f"Sin snapshots en las últimas {window_hours}h. Ejecutar el poller primero."}

        demands = [r["demand_mw"] for r in rows if r["demand_mw"] is not None]
        renewables_pcts: list[float] = []
        balances: list[float] = []

        for row in rows:
            if not row["generation_json"] or not row["demand_mw"]:
                continue
            gen = json.loads(row["generation_json"])
            wind = gen.get("wind_mw", 0) or 0
            solar = gen.get("solar_pv_mw", 0) or 0
            solar_t = gen.get("solar_thermal_mw", 0) or 0
            total_gen = sum(v for v in gen.values() if v)
            demand = row["demand_mw"]
            renewables_pcts.append((wind + solar + solar_t) / demand * 100 if demand else 0)
            balances.append(total_gen - demand)

        import statistics

        def _stats(lst: list[float]) -> tuple[float, float]:
            if not lst:
                return 0.0, 0.0
            return statistics.mean(lst), statistics.stdev(lst) if len(lst) > 1 else 0.0

        d_mean, d_std = _stats(demands)
        r_mean, r_std = _stats(renewables_pcts)
        b_mean, b_std = _stats(balances)

        return Baseline(
            demand_mean_mw=round(d_mean, 2),
            demand_std_mw=round(d_std, 2),
            renewables_pct_mean=round(r_mean, 2),
            renewables_pct_std=round(r_std, 2),
            balance_mean_mw=round(b_mean, 2),
            balance_std_mw=round(b_std, 2),
            window_hours=window_hours,
            computed_at=datetime.now(timezone.utc),
        ).model_dump(mode="json")

    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def query_incidents(start: str, end: str) -> list[dict]:
    """Recupera incidentes registrados en un intervalo de tiempo.

    Args:
        start: ISO 8601 UTC, ej. '2025-04-28T00:00:00Z'
        end:   ISO 8601 UTC, ej. '2025-04-28T23:59:59Z'
    """
    try:
        with _conn() as con:
            rows = con.execute(
                "SELECT * FROM incidents WHERE start >= ? AND start <= ? ORDER BY start",
                (start, end),
            ).fetchall()

        results = []
        for r in rows:
            try:
                incident = Incident(
                    incident_id=r["incident_id"],
                    start=datetime.fromisoformat(r["start"]),
                    end=datetime.fromisoformat(r["end"]) if r["end"] else None,
                    severity=r["severity"],
                    description=r["description"] or "",
                    resolution=r["resolution"],
                    risk_score_at_onset=r["risk_score_at_onset"] or 0,
                )
                results.append(incident.model_dump(mode="json"))
            except Exception:
                continue
        return results
    except Exception as exc:
        return [{"error": str(exc)}]


@mcp.tool()
def get_risk_history(hours: int = 24) -> list[dict]:
    """Recupera el historial de risk scores de las últimas N horas.

    Args:
        hours: Horas hacia atrás (default 24)
    """
    try:
        cutoff = (
            datetime.now(timezone.utc) - timedelta(hours=hours)
        ).isoformat()

        with _conn() as con:
            rows = con.execute(
                "SELECT * FROM risk_scores WHERE timestamp >= ? ORDER BY timestamp",
                (cutoff,),
            ).fetchall()

        results = []
        for r in rows:
            try:
                score = RiskScore(
                    value=r["value"],
                    level=r["level"],
                    blackout_probability=r["blackout_probability"] or 0.0,
                    components=json.loads(r["components_json"] or "{}"),
                    timestamp=datetime.fromisoformat(r["timestamp"]),
                )
                results.append(score.model_dump(mode="json"))
            except Exception:
                continue
        return results
    except Exception as exc:
        return [{"error": str(exc)}]


# ---------------------------------------------------------------------------
# Estado de Markov — helpers _impl + tools
# ---------------------------------------------------------------------------

def _init_db_at(db_path) -> None:
    """Versión de _init_db parametrizada por ruta (para tests). Crea SOLO las tablas grid_states y lessons (no el esquema completo)."""
    con = sqlite3.connect(str(db_path))
    con.executescript("""
        CREATE TABLE IF NOT EXISTS grid_states (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp  TEXT    NOT NULL,
            state      INTEGER NOT NULL,
            created_at TEXT    DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_grid_states_ts ON grid_states(timestamp);

        CREATE TABLE IF NOT EXISTS lessons (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT    NOT NULL,
            markov_state  INTEGER NOT NULL,
            p_band        TEXT    NOT NULL,
            incident_type TEXT,
            incident_id   TEXT,
            source_agent  TEXT    NOT NULL,
            context       TEXT    NOT NULL,
            lesson        TEXT    NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_lessons_ts ON lessons(timestamp);
    """)
    con.commit()
    con.close()


def _store_grid_state_impl(timestamp: str, state: int, db_path) -> dict:
    ts_norm = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).isoformat()
    con = sqlite3.connect(str(db_path))
    try:
        con.execute(
            "INSERT INTO grid_states (timestamp, state) VALUES (?, ?)",
            (ts_norm, int(state)),
        )
        con.commit()
        return {"ok": True}
    finally:
        con.close()


def _get_state_transitions_impl(
    hours: int, db_path, now: str | None = None
) -> list[dict]:
    ref = (
        datetime.fromisoformat(now.replace("Z", "+00:00"))
        if now
        else datetime.now(timezone.utc)
    )
    cutoff = (ref - timedelta(hours=hours)).isoformat()
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            "SELECT timestamp, state FROM grid_states "
            "WHERE timestamp >= ? ORDER BY timestamp",
            (cutoff,),
        ).fetchall()
        return [
            {
                "timestamp": r["timestamp"],
                "state": r["state"],
                "label": STATE_LABELS.get(r["state"], "?"),
            }
            for r in rows
        ]
    finally:
        con.close()


@mcp.tool()
def store_grid_state(timestamp: str, state: int) -> dict:
    """Persiste el estado de Markov S0–S4 en un instante (UTC ISO 8601).

    Args:
        timestamp: Instante UTC en formato ISO 8601, ej. '2026-06-30T10:00:00Z'
        state:     Estado de Markov (0=normal, 1=estrés, 2=notable, 3=mayor, 4=apagón)
    """
    return _store_grid_state_impl(timestamp, state, db_path=_db_path())


@mcp.tool()
def get_state_transitions(hours: int = 24) -> list[dict]:
    """Historial de estados de las últimas N horas: [{timestamp, state, label}].

    Args:
        hours: Horas hacia atrás (default 24)
    """
    return _get_state_transitions_impl(hours, db_path=_db_path())


# ---------------------------------------------------------------------------
# Lecciones aprendidas — helpers _impl + tools
# ---------------------------------------------------------------------------

def _store_lesson_impl(lesson_json: str, db_path=None) -> dict:
    path = db_path if db_path is not None else _db_path()
    try:
        data = json.loads(lesson_json)
        lesson = Lesson.model_validate(data)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    con = sqlite3.connect(str(path))
    try:
        cur = con.execute(
            """INSERT INTO lessons
               (timestamp, markov_state, p_band, incident_type, incident_id,
                source_agent, context, lesson)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                lesson.timestamp, lesson.markov_state, lesson.p_band,
                lesson.incident_type, lesson.incident_id, lesson.source_agent,
                lesson.context, lesson.lesson,
            ),
        )
        con.commit()
        return {"ok": True, "id": cur.lastrowid}
    finally:
        con.close()


def _get_lessons_impl(markov_state: int = -1, p_band: str = "", limit: int = 10, db_path=None) -> list[dict]:
    path = db_path if db_path is not None else _db_path()
    limit = min(max(int(limit), 1), 50)

    query = "SELECT * FROM lessons WHERE 1=1"
    params: list = []
    if markov_state >= 0:
        query += " AND markov_state = ?"
        params.append(markov_state)
    if p_band:
        query += " AND p_band = ?"
        params.append(p_band)
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(query, params).fetchall()
        return [dict(row) for row in rows]
    finally:
        con.close()


@mcp.tool()
def store_lesson(lesson_json: str) -> dict:
    """Persiste una lección aprendida al cierre de un ciclo relevante.

    Args:
        lesson_json: Lesson serializada como JSON string
    """
    return _store_lesson_impl(lesson_json)


@mcp.tool()
def get_lessons(markov_state: int = -1, p_band: str = "", limit: int = 10) -> list[dict]:
    """Recupera lecciones aprendidas, opcionalmente filtradas.

    Args:
        markov_state: Filtra por estado Markov exacto (-1 = sin filtro)
        p_band: Filtra por banda de escalación (vacío = sin filtro)
        limit: Máximo de lecciones a devolver (recortado a [1, 50])
    """
    return _get_lessons_impl(markov_state, p_band, limit)


# Inicializar BD al importar el módulo
_init_db()

if __name__ == "__main__":
    mcp.run()
