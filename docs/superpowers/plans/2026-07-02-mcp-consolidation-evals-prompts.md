# Consolidación MCP (7→5) + Evals por agente + Prompts versionables — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidar los 7 MCP servers en 5 (grid-data, weather, memory, ml-models, knowledge-base) podando tools huérfanas y arreglando IDs ESIOS; reestructurar `eval/` por agente en tres ejes (trigger/output/trayectoria); extraer los prompts a `prompts/` versionable.

**Architecture:** Nuevo `grid_data_server.py` absorbe entso-e íntegro + `get_generation_mix` con IDs verificados; `markov_server` se pliega en `ml_models_server`; el poller pasa a solo-lectura (la tabla `snapshots` queda en manos exclusivas de `incremental_load.catch_up`). Los evals se organizan en `eval/agents/<agente>/{test_trigger,test_trajectory,test_output}.py` con dos niveles (offline default + `@pytest.mark.integration` con LLM real vía harness compartido). Los `_INSTRUCTION` se cargan desde `prompts/<agente>.md` manteniendo `as_static_instruction`.

**Tech Stack:** Python 3.14 · google-adk 2.3.0 · FastMCP · pytest 9 · SQLite · ChromaDB (no se toca).

**Spec:** `docs/superpowers/specs/2026-07-02-mcp-consolidation-evals-prompts-design.md`

## Global Constraints

- Trabajar sobre `main` en `C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project`. El shell vuelve al worktree tras cada comando: **prefijar TODO comando** con `cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && `.
- `PYTHONUTF8=1` en todos los comandos python (consola cp1252 rompe con Unicode).
- NUNCA leer `.env`, `*.key`, `secrets.*`, `credentials.*`. Tokens vía variables de entorno.
- NUNCA hardcodear modelos Gemini → `config/model_loader.py::get_model()`.
- Instrucciones de agentes SIEMPRE vía `agents/instruction_utils.as_static_instruction` (evita KeyError de templating ADK 2.3.0 con llaves literales).
- Suite offline (`python -m pytest -q`) debe seguir verde tras CADA task: hoy 205 passed, 7 deselected. `pytest.ini` mantiene `addopts = -m "not integration"`.
- Commits en español (conventional). Terminar cada mensaje con `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Los nombres de tools MCP NO cambian en ningún task (las whitelists de `guardrails/hooks.py::_ALLOWED_TOOLS` siguen válidas; único cambio permitido: quitar `store_snapshot` del orchestrator en Task 6).

---

## BLOQUE A — Consolidación MCP (Tasks 1–7)

### Task 1: Crear `grid_data_server.py` (entso-e + generation mix con IDs verificados)

**Files:**
- Create: `mcp_servers/grid_data_server.py`
- Create: `tests/mcp_servers/test_grid_data_server.py`
- Modify: `tests/mcp_servers/test_eu_market_prices.py:13`

**Interfaces:**
- Consumes: `schemas.models` (CapacityMargin, CrossBorderFlow, EUGenerationSnapshot, MarketPrice, GenerationMix), `ml/data/download_esios.py::INDICATORS` (solo en tests, como fuente de verdad de paridad).
- Produces: módulo `mcp_servers.grid_data_server` con tools MCP `get_generation_mix`, `get_cross_border_flows`, `get_eu_generation`, `get_capacity_margin`, `get_eu_market_prices`; expone `_GEN_INDICATORS: dict[str,int]`, `_market_prices_impl` y `AREA_CODES` (los usan los tests). Tasks 2 lo conecta a agentes.

- [ ] **Step 1: Escribir el test de paridad de IDs (failing)**

Crear `tests/mcp_servers/test_grid_data_server.py`:

```python
"""Tests del MCP grid-data: paridad de IDs ESIOS con download_esios (fuente verificada)."""
from __future__ import annotations


def test_generation_ids_match_verified_download_esios():
    """Los IDs de generación deben ser EXACTAMENTE los verificados en producción
    (ml/data/download_esios.py, junio 2026). El bug original: nuclear=460 → ~125 GW."""
    from mcp_servers import grid_data_server as gd
    from ml.data.download_esios import INDICATORS

    verified = INDICATORS["generation"]
    for key, ind_id in gd._GEN_INDICATORS.items():
        assert key in verified, f"{key} no existe en download_esios.INDICATORS['generation']"
        assert ind_id == verified[key], (
            f"ID divergente para {key}: grid_data={ind_id}, verificado={verified[key]}"
        )


def test_entsoe_tools_present():
    """Las 4 tools ENTSO-E migran íntegras desde entsoe_server."""
    from mcp_servers import grid_data_server as gd
    for fn in ("get_cross_border_flows", "get_eu_generation",
               "get_capacity_margin", "get_eu_market_prices", "get_generation_mix"):
        assert callable(getattr(gd, fn)), f"tool ausente: {fn}"


def test_area_codes_preserved():
    from mcp_servers import grid_data_server as gd
    assert set(gd.AREA_CODES) == {"ES", "FR", "PT"}
```

- [ ] **Step 2: Verificar que falla**

Run: `cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && set PYTHONUTF8=1 && python -m pytest tests/mcp_servers/test_grid_data_server.py -q`
Expected: FAIL/ERROR con `ModuleNotFoundError: No module named 'mcp_servers.grid_data_server'`

- [ ] **Step 3: Implementar `mcp_servers/grid_data_server.py`**

El fichero se compone de: (a) sección ESIOS nueva (abajo), (b) sección ENTSO-E copiada **verbatim** de `mcp_servers/entsoe_server.py` líneas 37–269 (desde `AREA_CODES = {` hasta el final de `get_eu_market_prices`, incluidas `_token`→renombrada, `_client`, `_now_window`, las 4 tools y `_market_prices_impl`/`_fetch_day_ahead`). Único cambio permitido en el bloque copiado: renombrar `_token` → `_entsoe_token` (y sus 1 uso en `_client`) y `_cached` → usar la versión con TTL explícito de abajo (5 call sites: añadir `, CACHE_TTL_ENTSOE` como tercer argumento).

```python
"""
mcp_servers/grid_data_server.py — MCP: grid-data

Datos de red externos en un único server: ESIOS (mix de generación español)
+ ENTSO-E (interconexión europea y precios day-ahead).
Consolida los antiguos ree_grid_server (solo la tool consumida) y entsoe_server
(íntegro). Credenciales: ESIOS_TOKEN y ENTSOE_TOKEN (cada tool usa la suya).
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")
from typing import Any

import pandas as pd
import requests
from mcp.server.fastmcp import FastMCP

from schemas.models import (
    CapacityMargin, CrossBorderFlow, EUGenerationSnapshot, GenerationMix, MarketPrice,
)

mcp = FastMCP("gridguard-grid-data")

ESIOS_BASE_URL = "https://api.esios.ree.es"
CACHE_TTL_ESIOS = 540    # 9 min — cadencia ESIOS
CACHE_TTL_ENTSOE = 3600  # 1 h — cadencia ENTSO-E

_cache: dict[str, tuple[Any, float]] = {}


def _cached(key: str, build_fn, ttl: int):
    mono = time.monotonic()
    if key in _cache and _cache[key][1] > mono:
        return _cache[key][0]
    result = build_fn()
    _cache[key] = (result, mono + ttl)
    return result


# ---------------------------------------------------------------------------
# Sección ESIOS — get_generation_mix
# IDs verificados contra producción (ml/data/download_esios.py, junio 2026).
# La paridad la vigila tests/mcp_servers/test_grid_data_server.py.
# ---------------------------------------------------------------------------

_GEN_INDICATORS = {
    "nuclear_mw": 1153, "hydro_mw": 10035, "wind_mw": 10037,
    "solar_thermal_mw": 1162, "solar_pv_mw": 1161,
    "gas_cc_mw": 1156, "coal_mw": 10036, "cogeneration_mw": 10039,
}


def _esios_token() -> str:
    t = os.environ.get("ESIOS_TOKEN")
    if not t:
        raise EnvironmentError("ESIOS_TOKEN no definida")
    return t


def _esios_headers() -> dict:
    # ESIOS exige x-api-key (verificado 2026-07-01 vía api-probe); el esquema
    # legacy `Authorization: Token token="..."` por sí solo devuelve 403.
    token = _esios_token()
    return {
        "Accept": "application/json; application/vnd.esios-api-v2+json",
        "Authorization": f'Token token="{token}"',
        "x-api-key": token,
    }


def _latest_value(values: list[dict]) -> float:
    if not values:
        return 0.0
    return float(sorted(values, key=lambda x: x["datetime"])[-1].get("value", 0) or 0)


@mcp.tool()
def get_generation_mix(start: str, end: str) -> dict:
    """Mix de generación por tecnología en un intervalo (MW).

    Args:
        start: ISO 8601 UTC, ej. '2025-04-28T10:00:00Z'
        end:   ISO 8601 UTC, ej. '2025-04-28T11:00:00Z'
    """
    key = f"genmix_{start}_{end}"

    def build():
        tech_mw: dict[str, float] = {}
        for col, ind_id in _GEN_INDICATORS.items():
            url = f"{ESIOS_BASE_URL}/indicators/{ind_id}"
            resp = requests.get(
                url, headers=_esios_headers(),
                params={"start_date": start, "end_date": end, "time_trunc": "hour"},
                timeout=30,
            )
            if resp.ok:
                vals = resp.json()["indicator"]["values"]
                tech_mw[col] = _latest_value(vals)

        total = sum(tech_mw.values())
        return GenerationMix(
            start=datetime.fromisoformat(start.replace("Z", "+00:00")),
            end=datetime.fromisoformat(end.replace("Z", "+00:00")),
            technology_mw=tech_mw,
            total_mw=total if total > 0 else 1.0,
        ).model_dump(mode="json")

    return _cached(key, build, CACHE_TTL_ESIOS)


# ---------------------------------------------------------------------------
# Sección ENTSO-E — copiada verbatim de entsoe_server.py líneas 37–269
# (AREA_CODES, _entsoe_token, _client, _now_window, get_cross_border_flows,
#  get_eu_generation, get_capacity_margin, _market_prices_impl,
#  _fetch_day_ahead, get_eu_market_prices), con `_cached(key, build)` →
#  `_cached(key, build, CACHE_TTL_ENTSOE)`.
# ---------------------------------------------------------------------------

# <<< pegar aquí el bloque copiado >>>


if __name__ == "__main__":
    mcp.run()
```

Nota `get_generation_mix`: la clave del dict `tech_mw` es directamente la del indicador (`nuclear_mw`, …), igual que producía el `tech_map` del server antiguo — el schema `GenerationMix.technology_mw` no cambia.

- [ ] **Step 4: Migrar el import de `test_eu_market_prices.py`**

En `tests/mcp_servers/test_eu_market_prices.py` línea 13:

```python
# antes
from mcp_servers import entsoe_server as es
# después
from mcp_servers import grid_data_server as es
```

- [ ] **Step 5: Verificar que pasa todo**

Run: `cd "..." && set PYTHONUTF8=1 && python -m pytest tests/mcp_servers/test_grid_data_server.py tests/mcp_servers/test_eu_market_prices.py -q`
Expected: PASS (todos).

- [ ] **Step 6: Suite completa y commit**

Run: `cd "..." && set PYTHONUTF8=1 && python -m pytest -q` → verde (208+ ahora).

```bash
git add mcp_servers/grid_data_server.py tests/mcp_servers/test_grid_data_server.py tests/mcp_servers/test_eu_market_prices.py
git commit -m "feat(mcp): nuevo grid_data_server (ENTSO-E + generation mix con IDs verificados)"
```

---

### Task 2: Conectar agentes a grid-data y retirar `ree_grid_server` + `entsoe_server`

**Files:**
- Modify: `agents/renewables_forecast.py:76-83` (`_ree_toolset`)
- Modify: `agents/cross_border.py:79-86` (`_entsoe_toolset`) y docstring líneas 9–12
- Delete: `mcp_servers/ree_grid_server.py`, `mcp_servers/entsoe_server.py`
- Create: `tests/agents/test_mcp_wiring.py`

**Interfaces:**
- Consumes: `mcp_servers.grid_data_server` (Task 1).
- Produces: `tests/agents/test_mcp_wiring.py` con helper `_agent_source(module_name) -> str` que reutilizan Tasks 4 y 10.

- [ ] **Step 1: Test de wiring (failing)**

Crear `tests/agents/test_mcp_wiring.py`:

```python
"""Wiring de MCP servers por agente: qué server conecta cada módulo (fuente: el código)."""
from __future__ import annotations

from pathlib import Path

_AGENTS_DIR = Path(__file__).resolve().parents[2] / "agents"


def _agent_source(module_name: str) -> str:
    return (_AGENTS_DIR / f"{module_name}.py").read_text(encoding="utf-8")


def test_renewables_uses_grid_data_not_ree_grid():
    src = _agent_source("renewables_forecast")
    assert "mcp_servers.grid_data_server" in src
    assert "mcp_servers.ree_grid_server" not in src


def test_cross_border_uses_grid_data_not_entsoe():
    src = _agent_source("cross_border")
    assert "mcp_servers.grid_data_server" in src
    assert "mcp_servers.entsoe_server" not in src


def test_retired_servers_gone():
    servers = Path(__file__).resolve().parents[2] / "mcp_servers"
    assert not (servers / "ree_grid_server.py").exists()
    assert not (servers / "entsoe_server.py").exists()
```

- [ ] **Step 2: Verificar que falla**

Run: `cd "..." && set PYTHONUTF8=1 && python -m pytest tests/agents/test_mcp_wiring.py -q`
Expected: FAIL en los 3 tests.

- [ ] **Step 3: Cambiar los toolsets**

En `agents/renewables_forecast.py` (bloque `_ree_toolset`, líneas 76–83): sustituir `args=["-m", "mcp_servers.ree_grid_server"]` por `args=["-m", "mcp_servers.grid_data_server"]` y renombrar la variable `_ree_toolset` → `_grid_data_toolset` (también en la lista `tools=[...]` línea 118).

En `agents/cross_border.py` (bloque `_entsoe_toolset`, líneas 79–86): sustituir `args=["-m", "mcp_servers.entsoe_server"]` por `args=["-m", "mcp_servers.grid_data_server"]` y renombrar `_entsoe_toolset` → `_grid_data_toolset` (también en `tools=[...]`). Actualizar el docstring (líneas 9–12): `(MCP: entsoe_server)` → `(MCP: grid_data_server)`.

- [ ] **Step 4: Borrar los servers retirados**

```bash
git rm mcp_servers/ree_grid_server.py mcp_servers/entsoe_server.py
```

- [ ] **Step 5: Suite completa**

Run: `cd "..." && set PYTHONUTF8=1 && python -m pytest -q`
Expected: verde. Si algún test viejo importa `ree_grid_server`/`entsoe_server` directamente, migrarlo a `grid_data_server` (mismo símbolo) — según la auditoría solo `test_eu_market_prices.py` importaba y ya se migró en Task 1.

- [ ] **Step 6: Commit**

```bash
git add agents/renewables_forecast.py agents/cross_border.py tests/agents/test_mcp_wiring.py
git commit -m "refactor(mcp): agentes conectan grid-data; retirados ree_grid_server y entsoe_server"
```

---

### Task 3: Fold de markov en ml-models (server + tests)

**Files:**
- Modify: `mcp_servers/ml_models_server.py` (añadir sección Markov al final, antes de `if __name__`)
- Modify: `tests/mcp_servers/test_markov_server.py` (migrar imports; eliminar tests de `get_transition_probabilities`)
- Delete: `mcp_servers/markov_server.py` (en Task 4, cuando ya nadie lo conecte)

**Interfaces:**
- Consumes: `tools/markov_state_tracker` (`compute_state`, `get_trajectory`, `get_transition_matrix`, `prob_reach_severity`), `ml/incidents` (`S3_MAJOR`, `S4_BLACKOUT`, `STATE_LABELS`).
- Produces: tools MCP `get_grid_state(features)` y `get_markov_trajectory(features, horizon_hours=6)` en `mcp_servers.ml_models_server`, con impls `_get_grid_state_impl` / `_get_markov_trajectory_impl` (mismas firmas y payloads que en markov_server).

- [ ] **Step 1: Migrar el test (failing)**

En `tests/mcp_servers/test_markov_server.py`: cambiar línea 4 `from mcp_servers import markov_server as srv` → `from mcp_servers import ml_models_server as srv`. Eliminar cualquier test que use `_get_transition_probabilities_impl` o `get_transition_probabilities` (tool huérfana que muere). Renombrar el fichero:

```bash
git mv tests/mcp_servers/test_markov_server.py tests/mcp_servers/test_markov_tools.py
```

- [ ] **Step 2: Verificar que falla**

Run: `cd "..." && set PYTHONUTF8=1 && python -m pytest tests/mcp_servers/test_markov_tools.py -q`
Expected: FAIL con `AttributeError` (ml_models_server no tiene `_get_grid_state_impl`).

- [ ] **Step 3: Añadir la sección Markov a `ml_models_server.py`**

Al final del fichero, justo antes de `if __name__ == "__main__":` (línea 557), añadir:

```python
# ---------------------------------------------------------------------------
# Capa Markov (absorbida de markov_server.py — inferencia determinista, sin LLM)
# ---------------------------------------------------------------------------

from ml.incidents import S3_MAJOR, S4_BLACKOUT, STATE_LABELS  # noqa: E402
from tools import markov_state_tracker as mst  # noqa: E402


def _get_grid_state_impl(features: dict) -> dict:
    state = mst.compute_state(features)
    traj = mst.get_trajectory(state, horizon_hours=6)
    A = mst.get_transition_matrix()
    confidence = float(A[state].max())  # masa de la transición dominante
    return {
        "state": int(state),
        "label": STATE_LABELS.get(state, "?"),
        "confidence": confidence,
        "trajectory_6h": [int(s) for s in traj],
    }


def _get_markov_trajectory_impl(features: dict, horizon_hours: int) -> dict:
    state = mst.compute_state(features)
    return {
        "trajectory": [int(s) for s in mst.get_trajectory(state, horizon_hours)],
        "prob_reach_s3": float(mst.prob_reach_severity(state, S3_MAJOR, horizon_hours)),
        "prob_reach_s4": float(mst.prob_reach_severity(state, S4_BLACKOUT, horizon_hours)),
    }


@mcp.tool()
def get_grid_state(features: dict) -> dict:
    """Estado actual S0–S4 + trayectoria probable a 6 h desde los proxies de red."""
    return _get_grid_state_impl(features)


@mcp.tool()
def get_markov_trajectory(features: dict, horizon_hours: int = 6) -> dict:
    """Trayectoria de estado proyectada y P(alcanzar S3/S4) en el horizonte dado."""
    return _get_markov_trajectory_impl(features, horizon_hours)
```

Actualizar también el docstring de cabecera del módulo (líneas 13–17) añadiendo las dos tools nuevas a la lista de "Herramientas expuestas".

- [ ] **Step 4: Verificar que pasa**

Run: `cd "..." && set PYTHONUTF8=1 && python -m pytest tests/mcp_servers/test_markov_tools.py -q`
Expected: PASS.

- [ ] **Step 5: Suite completa y commit**

Run: `cd "..." && set PYTHONUTF8=1 && python -m pytest -q` → verde.

```bash
git add mcp_servers/ml_models_server.py tests/mcp_servers/test_markov_tools.py
git commit -m "refactor(mcp): capa Markov absorbida en ml_models_server (get_grid_state, get_markov_trajectory)"
```

---

### Task 4: Desconectar markov de los agentes y retirar `markov_server.py`

**Files:**
- Modify: `agents/orchestrator.py:178-185,201` · `agents/anomaly_detector.py:100-107` · `agents/renewables_forecast.py:103-110,121` · `agents/report_agent.py:167-174` · `agents/regulatory_compliance.py:53-60`
- Modify: `tests/agents/test_mcp_wiring.py` (añadir asserts)
- Delete: `mcp_servers/markov_server.py`

**Interfaces:**
- Consumes: tools Markov ya disponibles en `ml_models_server` (Task 3); helper `_agent_source` de Task 2.
- Produces: 6 agentes sin `_markov_toolset`; `regulatory_compliance` con `_ml_toolset` nuevo.

- [ ] **Step 1: Ampliar test de wiring (failing)**

Añadir a `tests/agents/test_mcp_wiring.py`:

```python
def test_markov_server_retired_from_all_agents():
    for mod in ("orchestrator", "anomaly_detector", "renewables_forecast",
                "regulatory_compliance", "report_agent"):
        assert "mcp_servers.markov_server" not in _agent_source(mod), mod


def test_regulatory_now_connects_ml_models():
    src = _agent_source("regulatory_compliance")
    assert "mcp_servers.ml_models_server" in src


def test_markov_server_file_gone():
    servers = Path(__file__).resolve().parents[2] / "mcp_servers"
    assert not (servers / "markov_server.py").exists()
```

Run: `... python -m pytest tests/agents/test_mcp_wiring.py -q` → FAIL en los 3 nuevos.

- [ ] **Step 2: Editar los 5 agentes**

- `orchestrator.py`, `anomaly_detector.py`, `renewables_forecast.py`, `report_agent.py`: **eliminar** el bloque `_markov_toolset = McpToolset(...)` completo y su entrada en la lista `tools=[...]` (estos 4 ya tienen `_ml_toolset`, que ahora sirve las tools Markov).
- `regulatory_compliance.py`: en su bloque `McpToolset` de markov (líneas 53–60), sustituir `args=["-m", "mcp_servers.markov_server"]` por `args=["-m", "mcp_servers.ml_models_server"]` y renombrar la variable `_markov_toolset` → `_ml_toolset` (también en `tools=[...]`).
- Revisar comentarios adyacentes (p. ej. `# Capa 2: MCP Tools (ML inference + memoria + Markov)` en orchestrator línea 198 → `# Capa 2: MCP Tools (ML inference + memoria)` no: Markov sigue, vía ml-models — dejar `(ML inference + Markov + memoria)`).

- [ ] **Step 3: Borrar el server**

```bash
git rm mcp_servers/markov_server.py
```

- [ ] **Step 4: Suite completa**

Run: `cd "..." && set PYTHONUTF8=1 && python -m pytest -q` → verde (ningún test importa ya `markov_server`; Task 3 migró el suyo).

- [ ] **Step 5: Commit**

```bash
git add agents/ tests/agents/test_mcp_wiring.py
git commit -m "refactor(agents): toolset markov sustituido por ml-models; retirado markov_server"
```

---

### Task 5: Podar `get_aemet_alerts` de weather

**Files:**
- Modify: `mcp_servers/weather_server.py:200-~250` (tool completa `get_aemet_alerts`)
- Modify: `tests/agents/test_mcp_wiring.py`

**Interfaces:**
- Consumes: nada nuevo.
- Produces: `weather_server` con exactamente 3 tools (`get_wind_forecast`, `get_solar_radiation`, `get_weather_forecast_24h`).

- [ ] **Step 1: Test (failing)**

Añadir a `tests/agents/test_mcp_wiring.py`:

```python
def test_weather_server_has_no_orphan_alerts_tool():
    servers = Path(__file__).resolve().parents[2] / "mcp_servers"
    src = (servers / "weather_server.py").read_text(encoding="utf-8")
    assert "get_aemet_alerts" not in src
```

Run → FAIL.

- [ ] **Step 2: Eliminar la tool**

En `mcp_servers/weather_server.py`: borrar el bloque completo de `get_aemet_alerts` (desde el `@mcp.tool()` de la línea 200 hasta el final de su función, antes de la siguiente definición). Si `MeteoAlert` queda sin uso en el fichero, quitar ese import (el schema en `schemas/models.py` NO se borra). Actualizar el docstring de cabecera del módulo (quitar la línea de `get_aemet_alerts`).

- [ ] **Step 3: Suite y commit**

Run: `cd "..." && set PYTHONUTF8=1 && python -m pytest -q` → verde (`tests/mcp_servers/test_weather_forecast_24h.py` no toca alerts).

```bash
git add mcp_servers/weather_server.py tests/agents/test_mcp_wiring.py
git commit -m "refactor(mcp): retirada get_aemet_alerts (huerfana) de weather_server"
```

---

### Task 6: Poller solo lectura — IDs verificados + retirar `store_snapshot`

**Files:**
- Modify: `tools/ree_poller.py:26-33` (`_GEN_INDICATORS`, `_INTERCHANGE_INDICATORS`)
- Modify: `guardrails/hooks.py:115` (whitelist orchestrator)
- Modify: `agents/orchestrator.py:29,86` (instruction paso 1g y docstring)
- Modify: `mcp_servers/memory_server.py:113-139` (tool `store_snapshot`)
- Create: `tests/tools/test_ree_poller_ids.py`

**Interfaces:**
- Consumes: `ml.data.download_esios.INDICATORS` (fuente de paridad en tests).
- Produces: `ree_poller` con IDs verificados; orchestrator sin `store_snapshot`; tabla `snapshots` con único escritor (`incremental_load.catch_up`).

- [ ] **Step 1: Test de paridad de IDs del poller (failing)**

Crear `tests/tools/test_ree_poller_ids.py`:

```python
"""Paridad de IDs ESIOS del poller con download_esios (fuente verificada en producción)."""
from __future__ import annotations

from ml.data.download_esios import INDICATORS
from tools import ree_poller as rp


def test_generation_ids_verified():
    for key, ind_id in rp._GEN_INDICATORS.items():
        assert ind_id == INDICATORS["generation"][key], (
            f"{key}: poller={ind_id}, verificado={INDICATORS['generation'][key]}"
        )


def test_interchange_ids_verified():
    for key, ind_id in rp._INTERCHANGE_INDICATORS.items():
        assert ind_id == INDICATORS["interchange"][key], (
            f"{key}: poller={ind_id}, verificado={INDICATORS['interchange'][key]}"
        )


def test_demand_real_id_verified():
    assert rp._DEMAND_INDICATORS["demand_real_mw"] == INDICATORS["demand"]["demand_real_mw"]
```

Run → FAIL (nuclear 460≠1153, es_fr 10209≠10045, …).

- [ ] **Step 2: Corregir IDs en `tools/ree_poller.py`**

Sustituir líneas 26–33 por:

```python
# IDs verificados contra producción (ml/data/download_esios.py, junio 2026).
# Paridad vigilada por tests/tools/test_ree_poller_ids.py.
_GEN_INDICATORS = {
    "nuclear_mw": 1153, "hydro_mw": 10035, "wind_mw": 10037,
    "solar_thermal_mw": 1162, "solar_pv_mw": 1161,
    "gas_cc_mw": 1156, "coal_mw": 10036, "cogeneration_mw": 10039,
}
_INTERCHANGE_INDICATORS = {
    "es_fr_mw": 10045, "es_pt_mw": 10044, "es_ma_mw": 10046,
}
```

`_DEMAND_INDICATORS` (1293/10034) y `_RESERVE_INDICATORS` (785) NO se tocan: 1293 está verificado; 10034 y 785 no tienen sustituto verificado en download_esios (dejar comentario `# sin ID verificado alternativo — mantener` sobre `_RESERVE_INDICATORS`). Nota semántica: los saldos 10045/10044/10046 son positivos = importación hacia ES, igual que asumía el consumidor (`stability_proxy_calculator`).

Run test → PASS.

- [ ] **Step 3: Quitar `store_snapshot` del orchestrator**

- `guardrails/hooks.py` línea 115: `"get_blackout_probability", "store_snapshot", "store_risk_score",` → `"get_blackout_probability", "store_risk_score",`.
- `agents/orchestrator.py` línea 86: `1g. Llama a `store_snapshot` y `store_risk_score` para persistir los datos.` → `1g. Llama a `store_risk_score` para persistir el risk score. (Los snapshots los persiste el pipeline incremental_load, NO este agente.)`.
- `agents/orchestrator.py` línea 29 (docstring módulo): `5. Persistencia en SQLite (store_snapshot, store_risk_score)` → `5. Persistencia del risk score en SQLite (store_risk_score)`.

- [ ] **Step 4: Retirar la tool `store_snapshot` de `memory_server.py`**

Auditoría previa (ya hecha en la spec, reconfirmar aquí): `grep -rn "store_snapshot" --include="*.py"` fuera de docs debe devolver solo `memory_server.py` tras el paso 3. Confirmado → borrar el bloque `@mcp.tool() def store_snapshot(...)` (líneas 113–139) y su línea en el docstring de cabecera. Si apareciera un consumidor nuevo, conservar la tool y anotar en el commit.

- [ ] **Step 5: Suite completa y commit**

Run: `cd "..." && set PYTHONUTF8=1 && python -m pytest -q` → verde (los tests de guardrails `test_allowed_tools.py:43` comprueban un subconjunto que no incluye `store_snapshot`).

```bash
git add tools/ree_poller.py guardrails/hooks.py agents/orchestrator.py mcp_servers/memory_server.py tests/tools/test_ree_poller_ids.py
git commit -m "fix(datos): poller con IDs ESIOS verificados y solo-lectura; snapshots con escritor unico"
```

---

### Task 7: Docs del bloque A (ARCHITECTURE, diagrama) + verificación

**Files:**
- Modify: `ARCHITECTURE.md` §7 (líneas 260–346), §13 árbol (545–559), línea 627, línea 220 y 296
- Modify: `architecture_diagram.html` (cards de servers; líneas 322, 335, 391, 664)

**Interfaces:** ninguna de código; cierre documental del bloque A.

- [ ] **Step 1: ARCHITECTURE.md §7**

Reescribir la sección `## 7. Layer 2 — MCP Servers (7 servers)` → `(5 servers)`:
- Card `MCP: ree-grid` y `MCP: entso-e` → una sola card `MCP: grid-data` (tools: `get_generation_mix`, `get_cross_border_flows`, `get_eu_generation`, `get_capacity_margin`, `get_eu_market_prices`; credenciales `ESIOS_TOKEN` + `ENTSOE_TOKEN`).
- Card `MCP: weather`: quitar `get_aemet_alerts`.
- Card `MCP: memory`: quitar `store_snapshot`.
- Card `MCP: ml-models`: añadir `get_grid_state`, `get_markov_trajectory` (y `get_ensemble_score`/`get_serve_features`/`get_blackout_probability_ensemble`/`reload_models` si no están listados).
- Card `MCP: markov`: eliminar entera.
- Línea 220 (tabla cron): `Invokes ree_poller() → stores snapshot` → `Invokes incremental_load.catch_up() → stores snapshots (training scale)`.
- Línea 627: `7 custom MCP servers (ree-grid, weather, entso-e, memory, ml-models, knowledge-base, markov)` → `5 custom MCP servers (grid-data, weather, memory, ml-models, knowledge-base)`.
- §13 árbol (545–559): reflejar ficheros reales (`grid_data_server.py`; sin `ree_grid_server.py`, `entsoe_server.py`, `markov_server.py`).

- [ ] **Step 2: architecture_diagram.html**

Buscar todas las menciones de los servers retirados (`ree-grid`, `entso-e`, `markov` como server, `store_snapshot`) y actualizarlas igual que §7. Mínimo: líneas 322/335 (poller ya no "→ snapshot"), 391 (card memory sin `store_snapshot()`), 664 y las cards de servers.

- [ ] **Step 3: Verificación de coherencia y commit**

Run: `cd "..." && grep -rn "markov_server\|entsoe_server\|ree_grid_server\|get_aemet_alerts\|store_snapshot" --include="*.py" .` (excluyendo docs/ y __pycache__) → 0 resultados en código fuente.
Run: `cd "..." && set PYTHONUTF8=1 && python -m pytest -q` → verde.

```bash
git add ARCHITECTURE.md architecture_diagram.html
git commit -m "docs(arquitectura): 5 MCP servers tras la consolidacion (grid-data, fold markov, poda)"
```

---

## BLOQUE B — Evals por agente (Tasks 8–12)

### Task 8: Scaffold `eval/agents/` + reparto de tests existentes + markers

**Files:**
- Create: `eval/agents/__init__.py` y `eval/agents/<agente>/__init__.py` para los 6: `orchestrator`, `anomaly_detector`, `renewables_forecast`, `regulatory_compliance`, `report_agent`, `cross_border`
- Create: `eval/conftest.py` (fixtures compartidas)
- Create: `eval/agents/orchestrator/test_output.py`, `eval/agents/orchestrator/test_trajectory.py`
- Create: `eval/common/__init__.py`, `eval/common/test_guardrails.py`
- Delete: `eval/test_outputs.py`, `eval/test_trajectories.py`
- Modify: `pytest.ini`

**Interfaces:**
- Produces: markers `agent_orchestrator|agent_anomaly|agent_renewables|agent_regulatory|agent_report|agent_cross_border`; fixtures compartidas en `eval/conftest.py` (`sample_snapshot`, `critical_snapshot`, `normal_snapshot`, `blackout_scenario`, `normal_scenario`) que Tasks 10–11 reutilizan.

- [ ] **Step 1: Registrar markers en `pytest.ini`**

Añadir bajo `markers =`:

```ini
    agent_orchestrator: evals del Orchestrator
    agent_anomaly: evals del AnomalyDetector
    agent_renewables: evals del RenewablesForecast
    agent_regulatory: evals del RegulatoryCompliance
    agent_report: evals del ReportAgent
    agent_cross_border: evals del CrossBorder
```

- [ ] **Step 2: Mover fixtures a `eval/conftest.py`**

Crear `eval/conftest.py` moviendo allí las fixtures módulo-nivel de `eval/test_outputs.py` (la fixture de la línea 53, `sample_snapshot`) y las de `eval/test_trajectories.py` (`critical_snapshot`, `normal_snapshot`, `blackout_scenario`, `normal_scenario` — copiar el cuerpo exacto tal como está en los ficheros de origen, junto con los imports que necesiten). Las fixtures conservan nombre y contenido: los tests movidos no cambian.

- [ ] **Step 3: Repartir los tests (git mv lógico)**

- `eval/test_outputs.py` clases `TestStabilityProxyCalculator`, `TestRiskScorer`, `TestAlertFormatter`, `TestDataNormalizer` → `eval/agents/orchestrator/test_output.py` (son las function tools del flujo del orchestrator). Añadir al principio del fichero: `pytestmark = pytest.mark.agent_orchestrator`.
- `eval/test_outputs.py` clase `TestGuardrails` → `eval/common/test_guardrails.py` (transversal, sin marker de agente).
- `eval/test_trajectories.py` íntegro (clases `TestBlackoutTrajectory`, `TestNormalTrajectory` + función `test_markov_trajectory_escalates_on_blackout_scenario`) → `eval/agents/orchestrator/test_trajectory.py` con `pytestmark = pytest.mark.agent_orchestrator` (los tests ya marcados `@pytest.mark.integration` conservan además ese marker).
- Borrar los dos ficheros de origen. Los imports de cada test viajan con él.

- [ ] **Step 4: Verificar conteo intacto**

Run: `cd "..." && set PYTHONUTF8=1 && python -m pytest -q`
Expected: mismo número de passed que antes del task (el reparto no añade ni quita tests) — comparar con la salida del Task 7.

Run: `cd "..." && set PYTHONUTF8=1 && python -m pytest eval/agents/orchestrator -q` → corre solo los del orchestrator.
Run: `cd "..." && set PYTHONUTF8=1 && python -m pytest -m agent_orchestrator -q` → ídem vía marker.

- [ ] **Step 5: Commit**

```bash
git add eval/ pytest.ini
git commit -m "refactor(eval): estructura por agente con markers; fixtures compartidas en conftest"
```

---

### Task 9: Harness live compartido (`eval/agents/_harness.py`)

**Files:**
- Create: `eval/agents/_harness.py`
- Create: `eval/agents/test_harness_smoke.py`

**Interfaces:**
- Consumes: `agents.orchestrator._extract_trajectory` y `_extract_result` (ya existen; reutilizar, NO copiar).
- Produces: `run_agent_live(agent, message_text: str) -> dict` con claves `output` (dict) y `trajectory` (list[dict] con clave `tool`). Lo usan los tests integration de Task 11.

- [ ] **Step 1: Smoke test offline (failing)**

Crear `eval/agents/test_harness_smoke.py`:

```python
"""El harness debe importar y exponer la firma esperada sin LLM (smoke offline)."""
import inspect


def test_harness_signature():
    from eval.agents import _harness
    sig = inspect.signature(_harness.run_agent_live)
    assert list(sig.parameters) == ["agent", "message_text"]
```

Run → FAIL (`ModuleNotFoundError`).

- [ ] **Step 2: Implementar `eval/agents/_harness.py`**

```python
"""Harness compartido para evals live (integration): ejecuta un Agent ADK
con un mensaje y devuelve output final + trayectoria de tools.

Generaliza el patrón de agents.orchestrator.run_scenario para poder evaluar
cualquier agente (incluidos los inner agents de las Skills) individualmente.
"""
from __future__ import annotations

import asyncio
from typing import Any

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from agents.orchestrator import _extract_result, _extract_trajectory


def run_agent_live(agent, message_text: str) -> dict[str, Any]:
    """Ejecuta `agent` (google.adk Agent) con `message_text` y devuelve
    {"output": dict, "trajectory": list[{"tool": str, ...}]}."""
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="gridguard_eval", session_service=session_service)
    session = asyncio.run(
        session_service.create_session(app_name="gridguard_eval", user_id="eval-runner")
    )
    message = genai_types.Content(role="user", parts=[genai_types.Part(text=message_text)])
    events = list(runner.run(session_id=session.id, user_id="eval-runner", new_message=message))
    return {"output": _extract_result(events), "trajectory": _extract_trajectory(events)}
```

Nota: los imports de `Runner`/`InMemorySessionService` deben calcarse de los que ya usa `agents/orchestrator.py` (mismas rutas de módulo ADK 2.3.0 — comprobarlas allí al implementar).

- [ ] **Step 3: Verificar y commit**

Run: `cd "..." && set PYTHONUTF8=1 && python -m pytest eval/agents/test_harness_smoke.py -q` → PASS.
Run: `cd "..." && set PYTHONUTF8=1 && python -m pytest -q` → verde.

```bash
git add eval/agents/_harness.py eval/agents/test_harness_smoke.py
git commit -m "feat(eval): harness live compartido para evaluar agentes individualmente"
```

---

### Task 10: Trigger offline por agente (wiring + whitelist coherentes)

**Files:**
- Create: `eval/agents/<agente>/test_trigger.py` para los 6 agentes

**Interfaces:**
- Consumes: `guardrails.hooks._ALLOWED_TOOLS`; código fuente de `agents/*.py`.
- Produces: cobertura trigger offline: para cada agente, el conjunto de tools que PUEDE disparar es exactamente el diseñado (whitelist) y los servers conectados son los que sirven esas tools.

- [ ] **Step 1: Escribir los 6 `test_trigger.py` (failing solo si hay incoherencia — se espera PASS directo; el valor es de regresión)**

Patrón (ejemplo completo para `eval/agents/cross_border/test_trigger.py`; replicar con su tabla para cada agente):

```python
"""Trigger offline del CrossBorderAgent: las tools que puede disparar son
exactamente las diseñadas, y sus servers conectados las sirven."""
from __future__ import annotations

from pathlib import Path

import pytest

from guardrails.hooks import _ALLOWED_TOOLS

pytestmark = pytest.mark.agent_cross_border

_EXPECTED_TOOLS = {
    "get_cross_border_flows", "get_capacity_margin",
    "get_eu_generation", "get_mitigation_procedure",
    "get_eu_market_prices",
}
_EXPECTED_SERVERS = {"mcp_servers.grid_data_server", "mcp_servers.knowledge_base_server"}


def test_whitelist_is_exactly_expected():
    assert _ALLOWED_TOOLS["cross_border_agent"] == _EXPECTED_TOOLS


def test_connected_servers_serve_the_whitelist():
    src = (Path(__file__).resolve().parents[3] / "agents" / "cross_border.py").read_text(encoding="utf-8")
    for server in _EXPECTED_SERVERS:
        assert server in src, f"server no conectado: {server}"
```

Tablas por agente (clave de whitelist → tools esperadas → servers esperados):

| whitelist key | `_EXPECTED_TOOLS` | `_EXPECTED_SERVERS` |
|---|---|---|
| `gridguard_orchestrator` | las 18 actuales de `hooks.py:112-120` **menos** `store_snapshot` | `ml_models_server`, `memory_server` |
| `anomaly_detector_skill` | las 7 de `hooks.py:121-125` | `ml_models_server`, `memory_server`, `knowledge_base_server`, `weather_server` |
| `renewables_forecast_skill` | las 7 de `hooks.py:126-130` | `weather_server`, `grid_data_server`, `ml_models_server`, `memory_server` |
| `report_agent` | las 7 de `hooks.py:131-135` | `memory_server`, `ml_models_server`, `knowledge_base_server` |
| `cross_border_agent` | las 5 de `hooks.py:136-140` | `grid_data_server`, `knowledge_base_server` |
| `regulatory_compliance_skill` | las 2 de `hooks.py:141-143` | `knowledge_base_server`, `ml_models_server` |

(Copiar los literales de tools desde `guardrails/hooks.py` al escribir cada tabla — el test debe listar el set explícito, no leerlo de hooks, para que una edición accidental de la whitelist falle el eval.)

- [ ] **Step 2: Verificar y commit**

Run: `cd "..." && set PYTHONUTF8=1 && python -m pytest eval/agents -q` → PASS todos.
Run: `cd "..." && set PYTHONUTF8=1 && python -m pytest -m agent_regulatory -q` → corre solo regulatory.

```bash
git add eval/agents
git commit -m "feat(eval): triggers offline por agente (whitelist y wiring exactos)"
```

---

### Task 11: Trigger + trayectoria live (integration) por agente

**Files:**
- Modify: `eval/agents/orchestrator/test_trigger.py` (añadir clase integration)
- Create: sección integration en `eval/agents/<agente>/test_trigger.py` de los otros 5

**Interfaces:**
- Consumes: `eval.agents._harness.run_agent_live` (Task 9); fixtures `blackout_scenario`/`normal_scenario` de `eval/conftest.py`; inner agents: `agents.anomaly_detector.AnomalyDetectorSkill.agent`, `agents.renewables_forecast.RenewablesForecastSkill.agent`, `agents.regulatory_compliance.RegulatoryComplianceSkill.agent`, `agents.report_agent.ReportAgent.agent`, `agents.cross_border.CrossBorderAgent.agent` (los `AgentTool` exponen `.agent`; verificar el nombre del atributo en ADK 2.3.0 — si fuera distinto, ajustar una sola vez aquí), y `agents.orchestrator.run_scenario` para el orchestrator.
- Produces: por agente, tests live que afirman trigger positivo, trigger negativo y orden de trayectoria.

- [ ] **Step 1: Orchestrator — triggers positivo y negativo**

Añadir a `eval/agents/orchestrator/test_trigger.py`:

```python
import pytest

from guardrails.hooks import _ALLOWED_TOOLS  # ya importado arriba

_SKILL_TOOLS = {"AnomalyDetectorSkill", "RenewablesForecastSkill",
                "RegulatoryComplianceSkill", "CrossBorderAgent", "ReportAgent"}


@pytest.mark.integration
class TestOrchestratorLiveTriggers:
    def test_critical_scenario_triggers_full_escalation(self, blackout_scenario):
        from agents.orchestrator import run_scenario
        result = run_scenario(blackout_scenario)
        tools = {step["tool"] for step in result["trajectory"]}
        missing = _SKILL_TOOLS - tools
        assert not missing, f"Escalación ROJA incompleta; faltan: {missing}"
        assert "alert_formatter" in tools

    def test_normal_scenario_triggers_nothing(self, normal_scenario):
        from agents.orchestrator import run_scenario
        result = run_scenario(normal_scenario)
        tools = {step["tool"] for step in result["trajectory"]}
        fired = _SKILL_TOOLS & tools
        assert not fired, f"Escenario VERDE no debe escalar; disparó: {fired}"
```

- [ ] **Step 2: Subagentes — plantilla live con tabla de expectativas**

Patrón (ejemplo completo para `eval/agents/report_agent/test_trigger.py`; replicar en los otros 4 con su tabla):

```python
import json

import pytest


@pytest.mark.integration
class TestReportAgentLiveTriggers:
    _PROMPT = (
        "[EVAL] Genera un informe de incidente para este estado de red critico. "
        "report_type='emergency'. Snapshot: "
        + json.dumps({"demand_mw": 30200, "renewables_pct": 74.0,
                      "blackout_probability": 0.72, "risk_score": 85})
    )

    def test_consults_history_before_writing(self):
        from agents.report_agent import ReportAgent
        from eval.agents._harness import run_agent_live
        result = run_agent_live(ReportAgent.agent, self._PROMPT)
        tools = [step["tool"] for step in result["trajectory"]]
        assert "find_similar_incident" in tools, "trigger RAG ausente"
        assert "get_markov_trajectory" in tools, "trigger Markov ausente"
        if "write_report" in tools:
            assert tools.index("find_similar_incident") < tools.index("write_report"), (
                "trayectoria invalida: write_report antes de consultar historicos"
            )
```

Tabla de expectativas live por agente (trigger positivo / negativo / orden):

| Agente | Debe disparar | No debe disparar | Orden exigido |
|---|---|---|---|
| anomaly_detector | `get_baseline`, `get_anomaly_score` | — | `get_baseline` antes de `get_anomaly_score` |
| renewables_forecast | `get_wind_forecast` o `get_solar_radiation`; `get_load_forecast` | — | tool meteo antes de la respuesta final (basta con que aparezca en la trayectoria) |
| regulatory_compliance | `search_regulations` | — | — |
| report_agent | `find_similar_incident`, `get_markov_trajectory` | — | históricos antes de `write_report` |
| cross_border (escenario margen bajo: prompt con `capacity_margin_pct: 3`) | `get_cross_border_flows`, `get_capacity_margin`, `get_mitigation_procedure` | — | — |
| cross_border (escenario holgado: `capacity_margin_pct: 35`, mode=assessment) | `get_cross_border_flows` | `get_mitigation_procedure` | — |

Prompts: construirlos como el `_PROMPT` del ejemplo, con prefijo `[EVAL]`, los campos de la tabla serializados con `json.dumps`, y sin llaves literales sueltas.

- [ ] **Step 3: Verificar recogida sin ejecutar LLM**

Run: `cd "..." && set PYTHONUTF8=1 && python -m pytest eval/agents --collect-only -q` → los integration aparecen recogidos y deseleccionados; `python -m pytest -q` sigue verde y con el mismo passed count + los offline nuevos.

(La ejecución live `pytest -m integration eval/agents/<agente>` requiere claves de API y NO forma parte del gate de este plan; queda documentada en Task 12.)

- [ ] **Step 4: Commit**

```bash
git add eval/agents
git commit -m "feat(eval): triggers y trayectoria live por agente (integration, harness compartido)"
```

---

### Task 12: README de eval con comandos por agente

**Files:**
- Create: `eval/README.md`

**Interfaces:** ninguna de código.

- [ ] **Step 1: Escribir `eval/README.md`**

```markdown
# Evals de GridGuard — por agente y por eje

Cada agente se evalúa en tres ejes: **trigger** (¿dispara las tools/skills
cuando debe y solo cuando debe?), **output** (¿el resultado es correcto?) y
**trayectoria** (¿el orden fue el correcto? shift-left Python→ML→LLM).

Dos niveles:
- **offline** (default, determinista, sin APIs): `python -m pytest eval -q`
- **live/integration** (LLM real + APIs; requiere claves): añadir `-m integration`

## Comandos por agente

| Agente | Offline | Live |
|---|---|---|
| orchestrator | `python -m pytest eval/agents/orchestrator -q` | `python -m pytest eval/agents/orchestrator -m integration` |
| anomaly_detector | `python -m pytest eval/agents/anomaly_detector -q` | `python -m pytest eval/agents/anomaly_detector -m integration` |
| renewables_forecast | `python -m pytest eval/agents/renewables_forecast -q` | `python -m pytest eval/agents/renewables_forecast -m integration` |
| regulatory_compliance | `python -m pytest eval/agents/regulatory_compliance -q` | `python -m pytest eval/agents/regulatory_compliance -m integration` |
| report_agent | `python -m pytest eval/agents/report_agent -q` | `python -m pytest eval/agents/report_agent -m integration` |
| cross_border | `python -m pytest eval/agents/cross_border -q` | `python -m pytest eval/agents/cross_border -m integration` |

También por marker: `python -m pytest -m agent_report -q` (markers en `pytest.ini`).

## Flujo prompt→eval

Los prompts viven en `prompts/<agente>.md`. Tras cambiar un prompt, corre el
eval de ese agente (offline y, si procede, live) y compara antes de commitear.

## Extensión futura

Framework nativo de eval de ADK (`.evalset.json` + `agents-cli eval`,
LLM-as-judge). No implementado: la suite actual es offline/determinista por
diseño (shift-left). Ver spec 2026-07-02.
```

- [ ] **Step 2: Commit**

```bash
git add eval/README.md
git commit -m "docs(eval): comandos por agente y guia de ejes trigger/output/trayectoria"
```

---

## BLOQUE C — Prompts versionables (Task 13)

### Task 13: Extraer `_INSTRUCTION` a `prompts/` con loader

**Files:**
- Create: `prompts/orchestrator.md`, `prompts/anomaly_detector.md`, `prompts/renewables_forecast.md`, `prompts/regulatory_compliance.md`, `prompts/report_agent.md`, `prompts/cross_border.md`
- Modify: `agents/instruction_utils.py` (añadir `load_prompt`)
- Modify: los 6 módulos de agente (sustituir el literal `_INSTRUCTION`)
- Create: `tests/agents/test_prompt_files.py`

**Interfaces:**
- Consumes: `as_static_instruction` (sin cambios).
- Produces: `load_prompt(agent_name: str) -> str` en `agents/instruction_utils.py`; convención `prompts/<nombre_modulo>.md`.

- [ ] **Step 1: Test (failing)**

Crear `tests/agents/test_prompt_files.py`:

```python
"""Prompts versionables: un fichero por agente, cargado en import, patron intacto."""
from __future__ import annotations

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_AGENTS = ["orchestrator", "anomaly_detector", "renewables_forecast",
           "regulatory_compliance", "report_agent", "cross_border"]


@pytest.mark.parametrize("name", _AGENTS)
def test_prompt_file_exists_and_non_empty(name):
    p = _ROOT / "prompts" / f"{name}.md"
    assert p.exists(), f"falta {p}"
    assert p.read_text(encoding="utf-8").strip(), f"prompt vacio: {p}"


@pytest.mark.parametrize("name", _AGENTS)
def test_agent_module_loads_prompt_from_file(name):
    src = (_ROOT / "agents" / f"{name}.py").read_text(encoding="utf-8")
    assert f'load_prompt("{name}")' in src, f"{name} no carga su prompt desde prompts/"
    assert '_INSTRUCTION = """' not in src, f"{name} conserva el literal inline"


def test_load_prompt_roundtrip():
    from agents.instruction_utils import load_prompt
    text = load_prompt("orchestrator")
    assert "PROTOCOLO DE MONITORIZACION" in text
    assert text == (_ROOT / "prompts" / "orchestrator.md").read_text(encoding="utf-8").strip()
```

Run → FAIL.

- [ ] **Step 2: Añadir `load_prompt` a `agents/instruction_utils.py`**

```python
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(agent_name: str) -> str:
    """Lee prompts/<agent_name>.md (UTF-8) y devuelve el texto listo para
    as_static_instruction. Falla alto y pronto si falta o esta vacio."""
    path = _PROMPTS_DIR / f"{agent_name}.md"
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt vacío: {path}")
    return text
```

- [ ] **Step 3: Extraer los 6 prompts**

Para cada módulo de agente (`agents/<name>.py`):
1. Copiar el contenido EXACTO del literal `_INSTRUCTION = """ ... """.strip()` (sin las comillas ni el `.strip()`) a `prompts/<name>.md`, byte a byte — las llaves literales `{...}` de los ejemplos JSON se conservan tal cual (el `.md` no se templetea nunca; `as_static_instruction` sigue haciendo el bypass).
2. Sustituir en el módulo:

```python
# antes
_INSTRUCTION = """
...texto...
""".strip()
# después
from agents.instruction_utils import as_static_instruction, load_prompt  # import ya existente, añadir load_prompt
_INSTRUCTION = load_prompt("orchestrator")  # ← nombre del módulo
```

La línea `instruction=as_static_instruction(_INSTRUCTION)` NO cambia en ningún agente.

Nota: en los módulos de Skills el literal es la instruction del inner agent (p. ej. `renewables_forecast.py` líneas 20–65); el fichero prompt lleva el nombre del MÓDULO (`renewables_forecast.md`), no del inner agent.

- [ ] **Step 4: Verificar**

Run: `cd "..." && set PYTHONUTF8=1 && python -m pytest tests/agents/test_prompt_files.py tests/agents/test_instruction_templating.py -q` → PASS (templating guard incluido).
Run: `cd "..." && set PYTHONUTF8=1 && python -m pytest -q` → verde.

- [ ] **Step 5: Commit**

```bash
git add prompts/ agents/ tests/agents/test_prompt_files.py
git commit -m "feat(prompts): instrucciones versionables en prompts/ con loader (patron as_static_instruction intacto)"
```

---

## Task 14: Cierre — ledger, memoria y push

**Files:**
- Modify: `.superpowers/sdd/progress.md` (nueva entrada Plan 7 con lo ejecutado)
- Modify: memoria del proyecto (`~/.claude/projects/<...>/memory/efficacy-ensemble-work.md` o fichero nuevo `mcp-consolidation.md` + línea en `MEMORY.md`)

- [ ] **Step 1: Verificación final completa**

Run: `cd "..." && set PYTHONUTF8=1 && python -m pytest -q` → verde, anotar el conteo final.
Run: `cd "..." && set PYTHONUTF8=1 && python -m pytest eval/agents/report_agent -q` → demuestra invocación individual.
Run: `cd "..." && grep -rn "markov_server\|entsoe_server\|ree_grid_server" --include="*.py" agents/ mcp_servers/ tools/ guardrails/ eval/ tests/` → 0 resultados.

- [ ] **Step 2: Actualizar ledger y memoria**

`progress.md`: entrada con fecha 2026-07-02, resumen de los 3 bloques, conteo de suite, y el backlog que siga vivo. Memoria: actualizar el estado (consolidación hecha, 5 servers, evals por agente, prompts extraídos; 4 temas RAG siguen `pending`).

- [ ] **Step 3: Push**

```bash
git push origin main
```

---

## Self-review del plan (hecho al escribirlo)

1. **Cobertura de spec:** Sección 1 → Tasks 1–7 (grid-data: 1–2; fold markov: 3–4; weather: 5; poller/store_snapshot: 6; docs: 7). Sección 2 → Tasks 8–12 (estructura+markers: 8; harness: 9; trigger offline: 10; trigger/trayectoria live: 11; comandos documentados: 12). Sección 3 → Task 13. Verificación/cierre → Task 14. Sin huecos.
2. **Placeholders:** el único bloque no transcrito es la sección ENTSO-E de Task 1, referida como copia verbatim con rango de líneas exacto (`entsoe_server.py:37-269`) y los 2 renombrados permitidos enumerados — decisión deliberada (menos error que transcribir 230 líneas).
3. **Consistencia de tipos/nombres:** `run_agent_live(agent, message_text)` (Task 9) == uso en Task 11; `load_prompt(name)` (Task 13) == asserts del test; `_GEN_INDICATORS` idéntico en Task 1 y Task 6; markers de Task 8 == `pytestmark` de Tasks 10–11.
