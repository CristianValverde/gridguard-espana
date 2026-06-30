# Plan 3 — Extensiones de servidores MCP

> **Para trabajadores agénticos:** SUB-SKILL REQUERIDO: `superpowers:subagent-driven-development` o
> `superpowers:executing-plans`. Lee el [índice maestro](2026-06-30-gridguard-incident-markov-agents-INDEX.md).
> Las Tasks 1–2 (weather/entsoe) son **independientes** y pueden hacerse en paralelo con Planes 1–2.
> Las Tasks 3–4 (memory/markov) requieren el [Plan 2](2026-06-30-plan2-markov-layer.md) completo.

**Objetivo:** dotar a la capa de agentes (Plan 4) de los datos que hoy no tiene: previsión meteorológica a
24 h, precios de mercado europeos, persistencia del estado de Markov y un MCP que exponga la trayectoria de
estado. Cuatro extensiones, dos de ellas contra APIs externas nuevas.

**Arquitectura:** se sigue el patrón de los servers existentes (`FastMCP`, caché TTL, tokens desde env). Para
que cada tool sea testeable sin depender de internals de FastMCP, **la lógica vive en una función privada
`_impl` y el `@mcp.tool()` la envuelve**; los tests prueban la `_impl`. Esto es una mejora de testabilidad
respecto a los servers actuales y se aplica a todo lo nuevo de este plan.

**Tech stack:** FastMCP (`mcp.server.fastmcp`), requests, entsoe-py, sqlite3, pydantic, pytest, responses
(mock HTTP).

## Constraints globales

Ver [índice maestro](2026-06-30-gridguard-incident-markov-agents-INDEX.md). **Crítico para este plan:**
- **Probe-first (CLAUDE.md global):** antes de escribir el código de integración de las Tasks 1 y 2, invocar
  el skill `api-probe`. Es el Step 1 de cada una. No se escribe parsing hasta que el probe valida el formato.
- Tokens (`AEMET_API_KEY`, `ENTSOE_TOKEN`) solo desde variables de entorno; nunca leer `.env` directamente
  (los servers ya hacen `load_dotenv`, que lee el fichero como entorno — eso es aceptable; lo prohibido es
  abrir y parsear `.env` con código propio).

---

## Estructura de ficheros

- **Modify** `mcp_servers/weather_server.py` — añadir `get_weather_forecast_24h` + `_forecast_24h_impl`.
- **Modify** `mcp_servers/entsoe_server.py` — añadir `get_eu_market_prices` + `_market_prices_impl`.
- **Modify** `mcp_servers/memory_server.py` — añadir tabla `grid_states`, `store_grid_state`,
  `get_state_transitions` (+ `_impl`).
- **Create** `mcp_servers/markov_server.py` — nuevo MCP que envuelve `tools/markov_state_tracker.py`.
- **Modify** `schemas/models.py` — nuevos schemas `MarketPrice`, `GridStateResult`, `StateHistoryEntry`.
- **Create** tests en `tests/mcp_servers/`.

> **Desviación justificada del handoff (DRY):** el handoff lista `get_state_history` en `markov_server` y
> `get_state_transitions` en `memory_server` — solapan. Se consolida la **persistencia** del historial en
> `memory_server` (SQLite) y se deja en `markov_server` solo el **cálculo puro** (estado, matriz, trayectoria).
> Así no hay dos lecturas del historial en sitios distintos.

---

### Task 1: `weather_server.get_weather_forecast_24h` (AEMET horaria, probe-first)

**Files:**
- Modify: `mcp_servers/weather_server.py`
- Modify: `schemas/models.py` (reutiliza `RadiationForecast`/`WindForecast`; no requiere schema nuevo)
- Test: `tests/mcp_servers/test_weather_forecast_24h.py`

**Interfaces:**
- Produce: `get_weather_forecast_24h(zones: list[str]) -> dict` → `{zone: {"timestamps": [...],
  "wind_speed_ms": [...], "radiation_wm2": [...], "forecast_hours": 24}}` para cada zona de
  `ZONE_COORDS` (`weather_server.py:43`). Internamente `_forecast_24h_impl(zones, fetch_fn)` separa la lógica
  del transporte HTTP para testear con un `fetch_fn` mockeado.

- [ ] **Step 1: PROBE OBLIGATORIO — invocar el skill `api-probe`**

Antes de escribir nada, invoca el skill `api-probe` para el endpoint AEMET de predicción horaria municipal:
`https://opendata.aemet.es/opendata/api/prediccion/especifica/municipio/horaria/{municipio}`.
Objetivo del probe: confirmar (a) el flujo de dos pasos de AEMET (la primera llamada devuelve una URL `datos`
que hay que volver a pedir), (b) la estructura JSON de la predicción horaria (campos de viento y, si existe,
de cielo/radiación), (c) el código de municipio para cada zona de `ZONE_COORDS`. Documenta el formato real
en el propio probe.

- [ ] **Step 2: Escribir el test de contrato que falla (respuesta mockeada)**

```python
# tests/mcp_servers/test_weather_forecast_24h.py
from mcp_servers import weather_server as ws


def _fake_fetch(zone, coords):
    # Forma confirmada por el probe; 24 entradas horarias por zona.
    return {
        "timestamps": [f"2026-06-30T{h:02d}:00:00Z" for h in range(24)],
        "wind_speed_ms": [5.0] * 24,
        "radiation_wm2": [0.0] * 6 + [300.0] * 12 + [0.0] * 6,
    }


def test_forecast_24h_returns_24h_per_zone():
    out = ws._forecast_24h_impl(["norte", "sur"], fetch_fn=_fake_fetch)
    assert set(out) == {"norte", "sur"}
    for zone_data in out.values():
        assert zone_data["forecast_hours"] == 24
        assert len(zone_data["timestamps"]) == 24
        assert len(zone_data["wind_speed_ms"]) == 24
        assert len(zone_data["radiation_wm2"]) == 24


def test_forecast_24h_unknown_zone_is_skipped():
    out = ws._forecast_24h_impl(["zona_inexistente"], fetch_fn=_fake_fetch)
    assert out == {}
```

- [ ] **Step 3: Ejecutar el test para verificar que falla**

Run: `python -m pytest tests/mcp_servers/test_weather_forecast_24h.py -v`
Expected: FAIL con `AttributeError: module 'mcp_servers.weather_server' has no attribute '_forecast_24h_impl'`.

- [ ] **Step 4: Implementar `_forecast_24h_impl` + el tool (parsing según el probe)**

```python
# mcp_servers/weather_server.py  (añadir)

def _forecast_24h_impl(zones: list[str], fetch_fn) -> dict:
    """Lógica pura: para cada zona conocida, obtiene 24 h de previsión vía fetch_fn.

    fetch_fn(zone, coords) -> {"timestamps": [...], "wind_speed_ms": [...], "radiation_wm2": [...]}
    El parsing real de la respuesta AEMET vive en _fetch_aemet_hourly (ajustado tras el probe).
    """
    out: dict = {}
    for zone in zones:
        coords = ZONE_COORDS.get(zone)
        if coords is None:
            continue
        data = fetch_fn(zone, coords)
        out[zone] = {
            "forecast_hours": 24,
            "timestamps": data["timestamps"][:24],
            "wind_speed_ms": data["wind_speed_ms"][:24],
            "radiation_wm2": data["radiation_wm2"][:24],
        }
    return out


def _fetch_aemet_hourly(zone: str, coords: dict) -> dict:
    """Transporte HTTP real a AEMET (flujo de 2 pasos confirmado por el probe).

    Implementar el parsing EXACTAMENTE según el formato documentado en el probe del Step 1.
    Usar AEMET_API_KEY desde os.environ; caché con CACHE_TTL como el resto del módulo;
    fallback a Open-Meteo (OPEN_METEO_BASE) si AEMET falla, como ya hace get_wind_forecast.
    """
    ...  # parsing según probe; respeta caché y fallback existentes


@mcp.tool()
def get_weather_forecast_24h(zones: list[str]) -> dict:
    """Previsión meteorológica horaria a 24 h para las zonas de generación renovable indicadas."""
    return _forecast_24h_impl(zones, fetch_fn=_fetch_aemet_hourly)
```

> El `...` de `_fetch_aemet_hourly` NO es un placeholder de diseño: el contrato (entrada/salida) está fijado y
> testeado vía `_forecast_24h_impl`; lo único pendiente es transcribir el parsing del formato que el probe
> documentó en el Step 1. Añade un test de integración real marcado `@pytest.mark.integration` que llame a
> AEMET una vez y verifique 24 entradas, para validar el parsing contra la API viva.

- [ ] **Step 5: Ejecutar los tests para verificar que pasan + commit**

Run: `python -m pytest tests/mcp_servers/test_weather_forecast_24h.py -v`
Expected: PASS (2 tests de contrato).

```bash
git add mcp_servers/weather_server.py tests/mcp_servers/test_weather_forecast_24h.py
git commit -m "feat(mcp): get_weather_forecast_24h en weather_server (AEMET horaria, probe-validado)"
```

---

### Task 2: `entsoe_server.get_eu_market_prices` (ENTSO-E A44 day-ahead, probe-first)

**Files:**
- Modify: `mcp_servers/entsoe_server.py`
- Modify: `schemas/models.py` (nuevo `MarketPrice`)
- Test: `tests/mcp_servers/test_eu_market_prices.py`

**Interfaces:**
- Produce: `get_eu_market_prices(countries: list[str], date: str) -> dict` → `{country: {"timestamps": [...],
  "price_eur_mwh": [...]}}`. `_market_prices_impl(countries, date, fetch_fn)` separa lógica del transporte.
- Nuevo schema `MarketPrice` en `schemas/models.py`.

- [ ] **Step 1: PROBE OBLIGATORIO — invocar el skill `api-probe`**

Invoca `api-probe` para el endpoint de precios day-ahead de ENTSO-E Transparency:
`https://web-api.tp.entsoe.eu/api?documentType=A44`. Confirma con el probe: (a) que `entsoe-py`
(`EntsoePandasClient.query_day_ahead_prices`, ya disponible vía `_client()` en `entsoe_server.py:51`) cubre
A44 sin construir la query a mano, (b) el formato de la Serie devuelta (índice temporal, €/MWh), (c) los
`AREA_CODES` necesarios (ya hay ES/FR/PT en `entsoe_server.py:37`).

- [ ] **Step 2: Escribir el test de contrato que falla**

```python
# tests/mcp_servers/test_eu_market_prices.py
import pandas as pd
from mcp_servers import entsoe_server as es


def _fake_fetch(country, day):
    idx = pd.date_range(f"{day} 00:00", periods=24, freq="h", tz="UTC")
    return pd.Series([50.0 + i for i in range(24)], index=idx)


def test_market_prices_returns_series_per_country():
    out = es._market_prices_impl(["FR", "PT"], "2026-06-30", fetch_fn=_fake_fetch)
    assert set(out) == {"FR", "PT"}
    for c in out.values():
        assert len(c["timestamps"]) == 24
        assert len(c["price_eur_mwh"]) == 24
        assert c["price_eur_mwh"][0] == 50.0


def test_market_prices_unknown_country_skipped():
    out = es._market_prices_impl(["XX"], "2026-06-30", fetch_fn=_fake_fetch)
    assert out == {}
```

- [ ] **Step 3: Ejecutar el test para verificar que falla**

Run: `python -m pytest tests/mcp_servers/test_eu_market_prices.py -v`
Expected: FAIL (`_market_prices_impl` no existe).

- [ ] **Step 4: Implementar el schema, `_market_prices_impl` y el tool**

En `schemas/models.py` (sección MCP entso-e, tras `CapacityMargin`, línea ~236):

```python
class MarketPrice(BaseModel):
    """Precio day-ahead por país. Salida de get_eu_market_prices()."""
    country: str
    timestamps: list[datetime]
    price_eur_mwh: list[float]
```

En `mcp_servers/entsoe_server.py`:

```python
from schemas.models import MarketPrice  # junto a los imports de schemas existentes


def _market_prices_impl(countries: list[str], date: str, fetch_fn) -> dict:
    """Lógica pura: precios day-ahead por país para la fecha dada (€/MWh)."""
    out: dict = {}
    for country in countries:
        if country not in AREA_CODES:
            continue
        series = fetch_fn(country, date)
        out[country] = MarketPrice(
            country=country,
            timestamps=list(series.index.to_pydatetime()),
            price_eur_mwh=[float(v) for v in series.values],
        ).model_dump(mode="json")
    return out


def _fetch_day_ahead(country: str, day: str):
    """Transporte real vía entsoe-py (query_day_ahead_prices). Ajustar tras el probe."""
    client = _client()
    start = pd.Timestamp(day, tz="Europe/Madrid")
    end = start + pd.Timedelta(days=1)
    return client.query_day_ahead_prices(country, start=start, end=end)


@mcp.tool()
def get_eu_market_prices(countries: list[str], date: str) -> dict:
    """Precios day-ahead (€/MWh) de los países indicados para la fecha (YYYY-MM-DD).

    Un spread ES-FR > 20 €/MWh sugiere excedente de generación en Francia (importación más fácil).
    """
    return _market_prices_impl(countries, date, fetch_fn=_fetch_day_ahead)
```

- [ ] **Step 5: Ejecutar los tests + commit**

Run: `python -m pytest tests/mcp_servers/test_eu_market_prices.py -v`
Expected: PASS.

```bash
git add mcp_servers/entsoe_server.py schemas/models.py tests/mcp_servers/test_eu_market_prices.py
git commit -m "feat(mcp): get_eu_market_prices en entsoe_server (A44 day-ahead, probe-validado)"
```

---

### Task 3: `memory_server` — persistencia del estado de Markov

**Files:**
- Modify: `mcp_servers/memory_server.py`
- Test: `tests/mcp_servers/test_memory_grid_states.py`

**Interfaces:**
- Produce:
  - `store_grid_state(timestamp: str, state: int) -> dict` → `{"ok": True}`; inserta en tabla `grid_states`.
  - `get_state_transitions(hours: int) -> list[dict]` → `[{"timestamp", "state", "label"}]` de las últimas
    N horas, ordenado. Usa `STATE_LABELS` de `ml.incidents`.
  - Funciones `_impl` testeables que aceptan un `db_path` para usar BD temporal en los tests.

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/mcp_servers/test_memory_grid_states.py
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
```

- [ ] **Step 2: Ejecutar el test para verificar que falla**

Run: `python -m pytest tests/mcp_servers/test_memory_grid_states.py -v`
Expected: FAIL (`_init_db_at`/`_store_grid_state_impl` no existen).

- [ ] **Step 3: Implementar tabla, helpers `_impl` y tools**

En `mcp_servers/memory_server.py`, añadir al `_init_db` (executescript, tras la tabla `incidents`,
línea ~96) la tabla nueva:

```python
            CREATE TABLE IF NOT EXISTS grid_states (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp  TEXT    NOT NULL,
                state      INTEGER NOT NULL,
                created_at TEXT    DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_grid_states_ts ON grid_states(timestamp);
```

Y añadir los helpers + tools (importando `STATE_LABELS` de `ml.incidents`):

```python
from datetime import datetime, timedelta, timezone
from ml.incidents import STATE_LABELS


def _init_db_at(db_path) -> None:
    """Versión de _init_db parametrizada por ruta (para tests)."""
    import sqlite3
    con = sqlite3.connect(str(db_path))
    con.executescript("""
        CREATE TABLE IF NOT EXISTS grid_states (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL, state INTEGER NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_grid_states_ts ON grid_states(timestamp);
    """)
    con.commit(); con.close()


def _store_grid_state_impl(timestamp: str, state: int, db_path) -> dict:
    import sqlite3
    con = sqlite3.connect(str(db_path))
    try:
        con.execute("INSERT INTO grid_states (timestamp, state) VALUES (?, ?)",
                    (timestamp, int(state)))
        con.commit()
        return {"ok": True}
    finally:
        con.close()


def _get_state_transitions_impl(hours: int, db_path, now: str | None = None) -> list[dict]:
    import sqlite3
    ref = datetime.fromisoformat(now.replace("Z", "+00:00")) if now else datetime.now(timezone.utc)
    cutoff = (ref - timedelta(hours=hours)).isoformat()
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            "SELECT timestamp, state FROM grid_states WHERE timestamp >= ? ORDER BY timestamp",
            (cutoff,),
        ).fetchall()
        return [{"timestamp": r["timestamp"], "state": r["state"],
                 "label": STATE_LABELS.get(r["state"], "?")} for r in rows]
    finally:
        con.close()


@mcp.tool()
def store_grid_state(timestamp: str, state: int) -> dict:
    """Persiste el estado de Markov S0–S4 en un instante (UTC ISO 8601)."""
    return _store_grid_state_impl(timestamp, state, db_path=_db_path())


@mcp.tool()
def get_state_transitions(hours: int = 24) -> list[dict]:
    """Historial de estados de las últimas N horas: [{timestamp, state, label}]."""
    return _get_state_transitions_impl(hours, db_path=_db_path())
```

- [ ] **Step 4: Ejecutar los tests + regresión del server + commit**

Run: `python -m pytest tests/mcp_servers/test_memory_grid_states.py -v`
Expected: PASS.

```bash
git add mcp_servers/memory_server.py tests/mcp_servers/test_memory_grid_states.py
git commit -m "feat(mcp): persistencia de estados de Markov en memory_server (grid_states)"
```

---

### Task 4: Nuevo MCP `markov_server.py`

**Files:**
- Create: `mcp_servers/markov_server.py`
- Modify: `schemas/models.py` (nuevos `GridStateResult`, `StateHistoryEntry`)
- Test: `tests/mcp_servers/test_markov_server.py`

**Interfaces:**
- Consume: `tools.markov_state_tracker` (Plan 2): `compute_state`, `get_transition_matrix`, `get_trajectory`,
  `prob_reach_severity`.
- Produce (tools MCP):
  - `get_grid_state(features: dict) -> dict` → `{state, label, confidence, trajectory_6h}`.
  - `get_transition_probabilities() -> dict` → `{matrix: list[list[float]], states: [0..4]}`.
  - `get_markov_trajectory(features: dict, horizon_hours: int) -> dict` →
    `{trajectory: list[int], prob_reach_s3: float, prob_reach_s4: float}`.

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/mcp_servers/test_markov_server.py
import numpy as np
import pytest
from mcp_servers import markov_server as srv


@pytest.fixture(autouse=True)
def _fake_model(tmp_path, monkeypatch):
    from ml import grid_state_tracker as gst
    from tools import markov_state_tracker as mst
    A = gst.blend_with_prior(np.eye(5) * 100)
    out = tmp_path / "markov_state.npz"
    np.savez(out, transition_matrix=A,
             thresholds=np.array({"p90_cp": .8, "p97_cp": .95, "p90_ren": 80., "p85_bal": 6000.}, dtype=object),
             state_counts=np.ones(5))
    monkeypatch.setattr(mst, "MODEL_PATH", out)
    mst._reset_cache()


def test_get_grid_state_shape():
    out = srv._get_grid_state_impl({"composite_pressure": 0.1, "renewables_pct": 30, "balance_mw": 0})
    assert set(out) == {"state", "label", "confidence", "trajectory_6h"}
    assert out["state"] == 0
    assert len(out["trajectory_6h"]) == 6


def test_transition_probabilities_shape():
    out = srv._get_transition_probabilities_impl()
    assert len(out["matrix"]) == 5 and len(out["matrix"][0]) == 5
    assert out["states"] == [0, 1, 2, 3, 4]


def test_markov_trajectory_probs_in_range():
    out = srv._get_markov_trajectory_impl(
        {"composite_pressure": 0.99, "renewables_pct": 90, "balance_mw": 9000}, horizon_hours=6)
    assert 0.0 <= out["prob_reach_s3"] <= 1.0
    assert 0.0 <= out["prob_reach_s4"] <= 1.0
    assert len(out["trajectory"]) == 6
```

- [ ] **Step 2: Ejecutar el test para verificar que falla**

Run: `python -m pytest tests/mcp_servers/test_markov_server.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'mcp_servers.markov_server'`.

- [ ] **Step 3: Implementar `markov_server.py`**

```python
"""
mcp_servers/markov_server.py — MCP: markov

Expone la capa de Markov (tools/markov_state_tracker.py) como herramientas MCP.
Pure Python, sin LLM (shift-left). Consumido por orchestrator, anomaly_detector,
report_agent y regulatory_compliance (Plan 4).
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from ml.incidents import S3_MAJOR, S4_BLACKOUT, STATE_LABELS
from tools import markov_state_tracker as mst

mcp = FastMCP("gridguard-markov")


def _get_grid_state_impl(features: dict) -> dict:
    state = mst.compute_state(features)
    traj = mst.get_trajectory(state, horizon_hours=6)
    A = mst.get_transition_matrix()
    confidence = float(A[state].max())  # masa de la transición dominante
    return {
        "state": state,
        "label": STATE_LABELS.get(state, "?"),
        "confidence": confidence,
        "trajectory_6h": traj,
    }


def _get_transition_probabilities_impl() -> dict:
    A = mst.get_transition_matrix()
    return {"matrix": [[float(x) for x in row] for row in A], "states": [0, 1, 2, 3, 4]}


def _get_markov_trajectory_impl(features: dict, horizon_hours: int) -> dict:
    state = mst.compute_state(features)
    return {
        "trajectory": mst.get_trajectory(state, horizon_hours),
        "prob_reach_s3": mst.prob_reach_severity(state, S3_MAJOR, horizon_hours),
        "prob_reach_s4": mst.prob_reach_severity(state, S4_BLACKOUT, horizon_hours),
    }


@mcp.tool()
def get_grid_state(features: dict) -> dict:
    """Estado actual S0–S4 + trayectoria probable a 6 h desde los proxies de red."""
    return _get_grid_state_impl(features)


@mcp.tool()
def get_transition_probabilities() -> dict:
    """Matriz de transición A[5x5] de la cadena de Markov."""
    return _get_transition_probabilities_impl()


@mcp.tool()
def get_markov_trajectory(features: dict, horizon_hours: int = 6) -> dict:
    """Trayectoria de estado proyectada y P(alcanzar S3/S4) en el horizonte dado."""
    return _get_markov_trajectory_impl(features, horizon_hours)


if __name__ == "__main__":
    mcp.run()
```

- [ ] **Step 4: Ejecutar los tests para verificar que pasan**

Run: `python -m pytest tests/mcp_servers/test_markov_server.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Smoke test de arranque del server + commit**

Run: `python -c "import mcp_servers.markov_server as m; print('tools OK')"`
Expected: imprime `tools OK` sin excepciones de import.

```bash
git add mcp_servers/markov_server.py schemas/models.py tests/mcp_servers/test_markov_server.py
git commit -m "feat(mcp): nuevo markov_server que expone estado y trayectoria de la red"
```

---

## Cierre del Plan 3

Al terminar: weather expone previsión a 24 h, entsoe expone precios day-ahead, memory persiste el historial de
estados y existe el MCP `markov_server`. **Prerrequisito cumplido para el Plan 4**, cuyos agentes consumen
`get_weather_forecast_24h`, `get_eu_market_prices`, `get_grid_state`, `get_markov_trajectory` y
`get_state_transitions`.

Run de regresión: `python -m pytest tests/mcp_servers/ -v`
Recuerda registrar `markov_server` en la configuración de arranque de MCP del proyecto (donde se declaran los
demás `StdioServerParameters`), para que los agentes del Plan 4 puedan conectarlo.
