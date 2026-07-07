"""grid_data: cota temporal dura por handler y caché que no envenena.

Regresión live 2026-07-03/04: un build de ENTSO-E tardó ~125s (2 queries +
parseo bs4) bloqueando el event loop del server; la siguiente request en
cola agotó los 120s del cliente MCP. Además, _cached guardaba también los
dicts {"error": ...} durante todo el TTL (1 h), envenenando el tool.
"""
from __future__ import annotations

import threading
import time

import pytest

from mcp_servers import grid_data_server as g


@pytest.fixture(autouse=True)
def _clean_cache():
    g._cache.clear()
    yield
    g._cache.clear()


def test_cached_acota_un_build_lento(monkeypatch):
    """Un build que excede la cota devuelve {"error": ...} en ~la cota,
    sin colgar el hilo llamante (que en producción es el event loop)."""
    monkeypatch.setattr(g, "BUILD_TIMEOUT_S", 0.2)
    SLEEP_S = 0.6  # 3x la cota: suficiente para disparar el timeout, sin dejar
    # el hilo huerfano (ThreadPoolExecutor no puede cancelar un hilo ya en
    # marcha) corriendo mucho tiempo tras el timeout.

    def _build_colgado():
        time.sleep(SLEEP_S)
        return {"ok": True}

    t0 = time.monotonic()
    result = g._cached("clave_lenta", _build_colgado, ttl=60)
    elapsed = time.monotonic() - t0

    assert isinstance(result, dict) and "error" in result, result
    assert elapsed < 5, f"la cota no actuó: {elapsed:.1f}s"

    # El future huerfano sigue vivo en _BUILD_POOL (pool global de proceso) tras
    # el timeout. Esperamos aqui a que termine para no dejar un hilo de fondo
    # corriendo durante los tests siguientes: causo un crash nativo real
    # (access violation) al colisionar con la carga de pyarrow.dataset en un
    # test posterior (test_train_classifier.py), reproducido 3/3 veces.
    time.sleep(SLEEP_S)


def test_cached_no_cachea_errores():
    calls = {"n": 0}

    def _build_falla_una_vez():
        calls["n"] += 1
        if calls["n"] == 1:
            return {"error": "ENTSO-E caído"}
        return {"ok": True}

    first = g._cached("clave_err", _build_falla_una_vez, ttl=3600)
    second = g._cached("clave_err", _build_falla_una_vez, ttl=3600)

    assert "error" in first
    assert second == {"ok": True}, "el error quedó cacheado y envenena el tool"


def test_cached_sigue_cacheando_resultados_buenos():
    calls = {"n": 0}

    def _build():
        calls["n"] += 1
        return {"ok": calls["n"]}

    first = g._cached("clave_ok", _build, ttl=3600)
    second = g._cached("clave_ok", _build, ttl=3600)
    assert first == second == {"ok": 1}
    assert calls["n"] == 1


def test_capacity_margin_consulta_generacion_y_demanda_en_paralelo(monkeypatch):
    """Las 2 queries ENTSO-E de get_capacity_margin deben ir en paralelo:
    secuenciales suman 2×30s y superan la cota del handler."""
    import pandas as pd

    barrier = threading.Barrier(2, timeout=5)

    class _FakeClient:
        def query_generation(self, country_code, start, end):
            barrier.wait()
            return pd.DataFrame({"wind": [100.0], "gas": [200.0]})

        def query_load(self, country_code, start, end):
            barrier.wait()
            return pd.Series([250.0])

    monkeypatch.setattr(g, "_client", lambda: _FakeClient())

    result = g.get_capacity_margin("ES")

    assert "error" not in result, result
    assert result["available_capacity_mw"] == 300.0
    assert result["peak_demand_mw"] == 250.0
