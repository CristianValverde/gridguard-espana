"""Tabla lessons + store_lesson/get_lessons: memoria de lecciones del orchestrator.

Spec: docs/superpowers/specs/2026-07-04-lessons-memory-design.md
"""
import json

from mcp_servers import memory_server as ms


def _lesson_json(**overrides) -> str:
    base = {
        "lesson": "Escalar a AMARILLO si Markov S2 pese a P baja.",
        "context": "P=0.08, Markov=S2 tras rampa solar.",
        "markov_state": 2,
        "p_band": "AMARILLO",
        "source_agent": "gridguard_orchestrator",
        "timestamp": "2026-07-04T10:00:00+00:00",
    }
    base.update(overrides)
    return json.dumps(base)


def test_store_and_get_lessons_roundtrip(tmp_path):
    db = tmp_path / "test.db"
    ms._init_db_at(db)

    result = ms._store_lesson_impl(_lesson_json(), db_path=db)
    assert result["ok"] is True
    assert isinstance(result["id"], int)

    rows = ms._get_lessons_impl(markov_state=-1, p_band="", limit=10, db_path=db)
    assert len(rows) == 1
    assert rows[0]["lesson"] == "Escalar a AMARILLO si Markov S2 pese a P baja."
    assert rows[0]["markov_state"] == 2
    assert rows[0]["p_band"] == "AMARILLO"


def test_get_lessons_filters_by_markov_state(tmp_path):
    db = tmp_path / "test.db"
    ms._init_db_at(db)
    ms._store_lesson_impl(_lesson_json(markov_state=1, p_band="AMARILLO"), db_path=db)
    ms._store_lesson_impl(_lesson_json(markov_state=3, p_band="NARANJA"), db_path=db)

    rows = ms._get_lessons_impl(markov_state=3, p_band="", limit=10, db_path=db)
    assert len(rows) == 1
    assert rows[0]["markov_state"] == 3


def test_get_lessons_filters_by_p_band(tmp_path):
    db = tmp_path / "test.db"
    ms._init_db_at(db)
    ms._store_lesson_impl(_lesson_json(markov_state=1, p_band="AMARILLO"), db_path=db)
    ms._store_lesson_impl(_lesson_json(markov_state=3, p_band="ROJO"), db_path=db)

    rows = ms._get_lessons_impl(markov_state=-1, p_band="ROJO", limit=10, db_path=db)
    assert len(rows) == 1
    assert rows[0]["p_band"] == "ROJO"


def test_get_lessons_orders_desc_by_timestamp(tmp_path):
    db = tmp_path / "test.db"
    ms._init_db_at(db)
    ms._store_lesson_impl(_lesson_json(timestamp="2026-07-01T00:00:00+00:00"), db_path=db)
    ms._store_lesson_impl(_lesson_json(timestamp="2026-07-03T00:00:00+00:00"), db_path=db)

    rows = ms._get_lessons_impl(markov_state=-1, p_band="", limit=10, db_path=db)
    assert rows[0]["timestamp"] == "2026-07-03T00:00:00+00:00"
    assert rows[1]["timestamp"] == "2026-07-01T00:00:00+00:00"


def test_get_lessons_limit_is_clamped(tmp_path):
    db = tmp_path / "test.db"
    ms._init_db_at(db)
    for i in range(5):
        ms._store_lesson_impl(_lesson_json(timestamp=f"2026-07-0{i+1}T00:00:00+00:00"), db_path=db)

    rows_over = ms._get_lessons_impl(markov_state=-1, p_band="", limit=999, db_path=db)
    assert len(rows_over) <= 50

    rows_under = ms._get_lessons_impl(markov_state=-1, p_band="", limit=0, db_path=db)
    assert len(rows_under) >= 1  # recortado a minimo 1, no 0


def test_store_lesson_invalid_json_returns_error_dict(tmp_path):
    db = tmp_path / "test.db"
    ms._init_db_at(db)

    result = ms._store_lesson_impl("{not valid json", db_path=db)
    assert result["ok"] is False
    assert "error" in result


def test_store_lesson_missing_required_field_returns_error_dict(tmp_path):
    db = tmp_path / "test.db"
    ms._init_db_at(db)

    bad = json.dumps({"context": "x", "markov_state": 1, "p_band": "AMARILLO"})  # falta 'lesson'
    result = ms._store_lesson_impl(bad, db_path=db)
    assert result["ok"] is False
    assert "error" in result


def test_get_lessons_empty_db_returns_empty_list(tmp_path):
    db = tmp_path / "test.db"
    ms._init_db_at(db)

    rows = ms._get_lessons_impl(markov_state=-1, p_band="", limit=10, db_path=db)
    assert rows == []
