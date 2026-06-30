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


def test_out_of_window_row_is_excluded(tmp_path):
    """El filtro de N horas debe excluir filas fuera de la ventana.

    Regresión contra el bug de comparación lexicográfica: timestamps
    terminados en 'Z' (ASCII 90) se comparaban como mayores que el cutoff
    terminado en '+' (ASCII 43), haciendo que el filtro fuera inerte.
    Tras la normalización a '+00:00' en _store_grid_state_impl ambos lados
    usan el mismo formato y la comparación es correcta.
    """
    db = tmp_path / "filter_test.db"
    ms._init_db_at(db)

    # now pinned a 2026-06-30T12:00:00Z
    now = "2026-06-30T12:00:00Z"

    # Fila DENTRO de la ventana: now - 1h → 11:00 (dentro de 24h)
    ms._store_grid_state_impl("2026-06-30T11:00:00Z", 1, db_path=db)
    # Fila FUERA de la ventana: now - 48h → 2026-06-28T12:00:00Z (fuera de 24h)
    ms._store_grid_state_impl("2026-06-28T12:00:00Z", 3, db_path=db)

    rows = ms._get_state_transitions_impl(hours=24, db_path=db, now=now)

    # Solo debe devolver la fila dentro de la ventana
    assert len(rows) == 1, (
        f"Se esperaba 1 fila dentro de la ventana, se obtuvieron {len(rows)}: {rows}"
    )
    assert rows[0]["state"] == 1
