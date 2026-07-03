"""El cliente ENTSO-E de grid_data debe estar acotado en tiempo.

Regresión live 2026-07-03: EntsoePandasClient por defecto usa
retry_count=3, retry_delay=10 y SIN timeout — una llamada colgada o con
retries bloquea el event loop del server más allá de los 60s del cliente
MCP y encola al resto de requests.
"""
from __future__ import annotations


def test_entsoe_client_con_timeout_y_sin_retries_largos(monkeypatch):
    captured: dict = {}

    import entsoe

    class _FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(entsoe, "EntsoePandasClient", _FakeClient)
    monkeypatch.setenv("ENTSOE_TOKEN", "token-test")

    from mcp_servers import grid_data_server as g

    g._client()

    timeout = captured.get("timeout")
    assert timeout is not None, "EntsoePandasClient sin timeout: llamada no acotada"
    assert 0 < timeout <= 30, f"timeout fuera de rango: {timeout}s"

    # retry_count=3 con retry_delay=10 por defecto = hasta +30s extra por llamada
    assert captured.get("retry_count", 3) <= 1, "retries largos no acotados"
    assert captured.get("retry_delay", 10) == 0, "retry_delay debe ser 0"
