from mcp_servers import memory_server as ms


def test_store_and_get_grid_states(tmp_path):
    db = tmp_path / "test.db"
    ms._init_db_at(db)  # crea esquema incl. grid_states
    ms._store_grid_state_impl("2026-06-30T10:00:00Z", 2, db_path=db)
    ms._store_grid_state_impl("2026-06-30T11:00:00Z", 1, db_path=db)

    rows = ms._get_state_transitions_impl(hours=24, db_path=db,
                                          now="2026-06-30T12:00:00Z")
    assert len(rows) == 2
    assert rows[0]["state"] == 2
    assert rows[0]["label"] == "Incidente notable"  # STATE_LABELS[2]
    assert rows[1]["state"] == 1
