"""tests/ml/conftest.py — apaga los thread pools globales de grid_data_server
antes de la suite de ML.

Hallazgo 2026-07-06: mcp_servers/grid_data_server.py crea dos ThreadPoolExecutor
a nivel de modulo (_BUILD_POOL, _QUERY_POOL) que nunca se cierran. Sus hilos
worker ("grid-build_N", "grid-query_N") quedan vivos (inactivos) el resto de la
sesion de pytest en cuanto cualquier test de tests/agents/ o tests/mcp_servers/
los usa. Cuando despues, en tests/ml/, pandas.read_parquet importa
pyarrow.dataset (extension nativa) con esos hilos todavia vivos, se reprodujo
un access violation real (varias veces seguidas, Windows, Python 3.14) que mata
el proceso entero -- crash reproducible con `pytest -q` (confirmado tambien en
PowerShell, no es un artefacto de ninguna herramienta en particular).

Por orden de directorio, tests/agents/ y tests/mcp_servers/ siempre terminan
antes de que arranque tests/ml/, asi que apagar los pools aqui (setup de esta
fixture, antes del primer test de ML) es seguro: ningun test posterior necesita
los pools ya vivos.
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def _shutdown_grid_data_thread_pools():
    try:
        from mcp_servers import grid_data_server as g
    except ImportError:
        yield
        return

    for pool_name in ("_BUILD_POOL", "_QUERY_POOL"):
        pool = getattr(g, pool_name, None)
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)

    yield
