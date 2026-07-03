# Toolsets MCP Singleton Compartidos — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compartir una única instancia de `McpToolset` por MCP server entre todos los agentes, reduciendo los subprocesos stdio de ~17 a 5 y desbloqueando los 2 evals live que nunca han pasado.

**Architecture:** Módulo nuevo `agents/mcp_toolsets.py` con 5 singletons; los 6 módulos de agentes eliminan sus 17 definiciones locales e importan los compartidos. El `MCPSessionManager` del ADK 2.3.0 cachea la sesión por instancia de toolset, así que compartir instancias = compartir subprocesos. Spec: `docs/superpowers/specs/2026-07-03-mcp-shared-transport-design.md`.

**Tech Stack:** Python 3.14 · google-adk 2.3.0 (`McpToolset`, `StdioConnectionParams`) · mcp (`StdioServerParameters`) · pytest.

## Global Constraints

- Trabajar sobre `main` del proyecto principal: prefijar cada comando con `cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && ` (el shell vuelve al worktree tras cada comando).
- `PYTHONUTF8=1` en todos los comandos python (la consola cp1252 rompe con Unicode).
- NUNCA leer `.env`, `*.key`, `secrets.*`, `credentials.*`; tokens solo por variables de entorno.
- NUNCA hardcodear nombres de modelos Gemini — siempre `config/model_loader.py::get_model()`.
- NO tocar el patrón `as_static_instruction` + `load_prompt` (evita el KeyError de templating de ADK 2.3.0), ni `register_hooks`, ni los prompts en `prompts/`.
- `pytest.ini` tiene `addopts = -m "not integration"`: `python -m pytest -q` corre solo la suite offline (243 passed, 16 deselected como línea base).
- Los tests `-m integration` consumen cuota Gemini real (~7-15 min por corrida): correr solo cuando el plan lo indica.
- Commits en español, conventional commits, con trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- El timeout de los `StdioConnectionParams` se mantiene en `60.0`.

---

### Task 1: Módulo `agents/mcp_toolsets.py` con los 5 singletons

**Files:**
- Create: `agents/mcp_toolsets.py`
- Test: `tests/agents/test_shared_toolsets.py`

**Interfaces:**
- Consumes: `McpToolset`, `StdioConnectionParams` (de `google.adk.tools.mcp_tool.mcp_toolset`), `StdioServerParameters` (de `mcp.client.stdio`) — mismos imports que usan hoy los agentes.
- Produces: constantes módulo-nivel `ML_TOOLSET`, `MEMORY_TOOLSET`, `KB_TOOLSET`, `WEATHER_TOOLSET`, `GRID_DATA_TOOLSET` (todas `McpToolset`), que la Task 2 importará desde los 6 módulos de agentes.

- [ ] **Step 1: Escribir los tests que fallan**

Crear `tests/agents/test_shared_toolsets.py`:

```python
"""Toolsets MCP singleton compartidos: 5 instancias, una por server.

Spec: docs/superpowers/specs/2026-07-03-mcp-shared-transport-design.md
"""
from __future__ import annotations

from google.adk.tools.mcp_tool.mcp_toolset import McpToolset

EXPECTED = {
    "ML_TOOLSET": "mcp_servers.ml_models_server",
    "MEMORY_TOOLSET": "mcp_servers.memory_server",
    "KB_TOOLSET": "mcp_servers.knowledge_base_server",
    "WEATHER_TOOLSET": "mcp_servers.weather_server",
    "GRID_DATA_TOOLSET": "mcp_servers.grid_data_server",
}


def test_expone_cinco_toolsets_mcp():
    from agents import mcp_toolsets

    for name in EXPECTED:
        assert isinstance(getattr(mcp_toolsets, name), McpToolset), name


def test_cada_toolset_apunta_a_su_server_con_timeout_60():
    from agents import mcp_toolsets

    for name, module in EXPECTED.items():
        toolset = getattr(mcp_toolsets, name)
        params = toolset._connection_params
        assert params.timeout == 60.0, name
        assert params.server_params.command == "python", name
        assert params.server_params.args == ["-m", module], name


def test_las_cinco_instancias_son_distintas():
    from agents import mcp_toolsets

    ids = {id(getattr(mcp_toolsets, name)) for name in EXPECTED}
    assert len(ids) == 5
```

Nota: `_connection_params` es atributo privado del ADK pero es la única superficie
donde el toolset expone sus parámetros; el test lo usa a sabiendas (si una futura
versión del ADK lo renombra, el test fallará ruidosamente, que es lo deseado).

- [ ] **Step 2: Verificar que fallan**

Run: `cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && PYTHONUTF8=1 python -m pytest tests/agents/test_shared_toolsets.py -v`
Expected: 3 FAILED / ERROR con `ModuleNotFoundError: No module named 'agents.mcp_toolsets'`

- [ ] **Step 3: Implementar el módulo**

Crear `agents/mcp_toolsets.py`:

```python
"""Toolsets MCP stdio compartidos (un singleton por server).

Todos los agentes importan estas 5 instancias en vez de crear las suyas: el
MCPSessionManager del ADK 2.3.0 cachea la sesion (y su subproceso) por
instancia de McpToolset, asi que compartir instancias reduce los subprocesos
de ~17 (uno por agente x server) a 5 (uno por server) y elimina la tormenta
de imports simultaneos de torch que hacia superar el timeout de 60s en los
evals live.

Spec: docs/superpowers/specs/2026-07-03-mcp-shared-transport-design.md
"""

from __future__ import annotations

from google.adk.tools.mcp_tool.mcp_toolset import McpToolset, StdioConnectionParams
from mcp.client.stdio import StdioServerParameters


def _stdio_toolset(module: str) -> McpToolset:
    return McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command="python",
                args=["-m", module],
            ),
            timeout=60.0,
        )
    )


ML_TOOLSET = _stdio_toolset("mcp_servers.ml_models_server")
MEMORY_TOOLSET = _stdio_toolset("mcp_servers.memory_server")
KB_TOOLSET = _stdio_toolset("mcp_servers.knowledge_base_server")
WEATHER_TOOLSET = _stdio_toolset("mcp_servers.weather_server")
GRID_DATA_TOOLSET = _stdio_toolset("mcp_servers.grid_data_server")
```

- [ ] **Step 4: Verificar que pasan**

Run: `cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && PYTHONUTF8=1 python -m pytest tests/agents/test_shared_toolsets.py -v`
Expected: 3 passed

- [ ] **Step 5: Suite offline completa**

Run: `cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && PYTHONUTF8=1 python -m pytest -q`
Expected: 246 passed, 16 deselected (243 previos + 3 nuevos)

- [ ] **Step 6: Commit**

```bash
git add agents/mcp_toolsets.py tests/agents/test_shared_toolsets.py
git commit -m "feat(mcp): modulo de toolsets MCP singleton compartidos (5 servers)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Migrar los 6 módulos de agentes a los toolsets compartidos

**Files:**
- Modify: `agents/orchestrator.py:46-86`
- Modify: `agents/anomaly_detector.py:24-84`
- Modify: `agents/renewables_forecast.py:24-82`
- Modify: `agents/regulatory_compliance.py:13-47`
- Modify: `agents/report_agent.py:29-132`
- Modify: `agents/cross_border.py:24-60`
- Modify: `tests/agents/test_mcp_wiring.py:13-39`
- Test: `tests/agents/test_shared_toolsets.py` (test nuevo de compartición global)

**Interfaces:**
- Consumes: `ML_TOOLSET`, `MEMORY_TOOLSET`, `KB_TOOLSET`, `WEATHER_TOOLSET`, `GRID_DATA_TOOLSET` de `agents.mcp_toolsets` (Task 1).
- Produces: los 6 agentes referencian exclusivamente los 5 singletons; ningún módulo de agente define `McpToolset` propio. Las Skills siguen exponiendo `.agent` (AgentTool) con su lista `tools`.

- [ ] **Step 1: Escribir el test de compartición global (falla antes de migrar)**

Añadir al final de `tests/agents/test_shared_toolsets.py`:

```python
def test_todos_los_agentes_comparten_los_singletons():
    from agents import mcp_toolsets
    from agents.anomaly_detector import AnomalyDetectorSkill
    from agents.cross_border import cross_border_agent
    from agents.orchestrator import orchestrator
    from agents.regulatory_compliance import RegulatoryComplianceSkill
    from agents.renewables_forecast import RenewablesForecastSkill
    from agents.report_agent import report_agent

    shared_ids = {id(getattr(mcp_toolsets, name)) for name in EXPECTED}
    agentes = [
        orchestrator,
        report_agent,
        cross_border_agent,
        AnomalyDetectorSkill.agent,
        RenewablesForecastSkill.agent,
        RegulatoryComplianceSkill.agent,
    ]
    encontrados = set()
    for agente in agentes:
        for tool in agente.tools:
            if isinstance(tool, McpToolset):
                assert id(tool) in shared_ids, (
                    f"{agente.name} usa un McpToolset propio en vez del compartido"
                )
                encontrados.add(id(tool))
    # Los 5 servers estan en uso por al menos un agente
    assert encontrados == shared_ids
```

- [ ] **Step 2: Verificar que falla**

Run: `cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && PYTHONUTF8=1 python -m pytest tests/agents/test_shared_toolsets.py::test_todos_los_agentes_comparten_los_singletons -v`
Expected: FAILED con `usa un McpToolset propio en vez del compartido`

- [ ] **Step 3: Migrar `agents/orchestrator.py`**

Reemplazar el import (línea 46):

```python
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset, StdioConnectionParams
```

y la línea 48 (`from mcp.client.stdio import StdioServerParameters`) — ambos se eliminan.
En el bloque de imports del proyecto, añadir:

```python
from agents.mcp_toolsets import MEMORY_TOOLSET, ML_TOOLSET
```

Eliminar los bloques `_ml_toolset = McpToolset(...)` (líneas 68-76) y
`_memory_toolset = McpToolset(...)` (líneas 78-86) completos.

En la lista `tools=[...]` del Agent, reemplazar:

```python
        _ml_toolset,
        _memory_toolset,
```

por:

```python
        ML_TOOLSET,
        MEMORY_TOOLSET,
```

Actualizar la nota de migración del docstring del módulo (línea 8):

```
- MCPToolset -> McpToolset; los toolsets stdio son singletons compartidos en
  agents/mcp_toolsets.py (un subproceso por server, no por agente x server).
```

- [ ] **Step 4: Migrar `agents/anomaly_detector.py`**

Eliminar los imports de las líneas 24 y 26 (`McpToolset, StdioConnectionParams` y
`StdioServerParameters`) y añadir:

```python
from agents.mcp_toolsets import KB_TOOLSET, MEMORY_TOOLSET, ML_TOOLSET, WEATHER_TOOLSET
```

Eliminar los 4 bloques `_ml_toolset/_memory_toolset/_kb_toolset/_weather_toolset = McpToolset(...)`
(líneas 35-73). En `tools=[...]` del `_inner_agent`, reemplazar:

```python
        _ml_toolset,
        _memory_toolset,
        _kb_toolset,
        _weather_toolset,
```

por:

```python
        ML_TOOLSET,
        MEMORY_TOOLSET,
        KB_TOOLSET,
        WEATHER_TOOLSET,
```

- [ ] **Step 5: Migrar `agents/renewables_forecast.py`**

Mismo patrón: eliminar imports de ADK/mcp para toolsets (líneas 24 y 26), añadir:

```python
from agents.mcp_toolsets import GRID_DATA_TOOLSET, MEMORY_TOOLSET, ML_TOOLSET, WEATHER_TOOLSET
```

Eliminar los 4 bloques de toolsets (líneas 34-72). En `tools=[...]`, reemplazar:

```python
        _weather_toolset,
        _grid_data_toolset,
        _ml_toolset,
        _memory_toolset,
```

por:

```python
        WEATHER_TOOLSET,
        GRID_DATA_TOOLSET,
        ML_TOOLSET,
        MEMORY_TOOLSET,
```

- [ ] **Step 6: Migrar `agents/regulatory_compliance.py`**

Eliminar imports de toolsets (líneas 13 y 15), añadir:

```python
from agents.mcp_toolsets import KB_TOOLSET, ML_TOOLSET
```

Eliminar los 2 bloques de toolsets (líneas 23-41). Reemplazar:

```python
    tools=[_kb_toolset, _ml_toolset],
```

por:

```python
    tools=[KB_TOOLSET, ML_TOOLSET],
```

- [ ] **Step 7: Migrar `agents/report_agent.py`**

Eliminar imports de toolsets (líneas 29 y 31), añadir:

```python
from agents.mcp_toolsets import KB_TOOLSET, MEMORY_TOOLSET, ML_TOOLSET
```

Eliminar los 3 bloques de toolsets (líneas 94-122). En `tools=[...]`, reemplazar:

```python
        _memory_toolset,
        _ml_toolset,
        _kb_toolset,
```

por:

```python
        MEMORY_TOOLSET,
        ML_TOOLSET,
        KB_TOOLSET,
```

- [ ] **Step 8: Migrar `agents/cross_border.py`**

Eliminar imports de toolsets (líneas 24 y 26), añadir:

```python
from agents.mcp_toolsets import GRID_DATA_TOOLSET, KB_TOOLSET
```

Eliminar los 2 bloques de toolsets (líneas 34-52). En `tools=[...]`, reemplazar:

```python
        _grid_data_toolset,
        _kb_toolset,
```

por:

```python
        GRID_DATA_TOOLSET,
        KB_TOOLSET,
```

- [ ] **Step 9: Actualizar `tests/agents/test_mcp_wiring.py`**

Estos 3 tests afirman sobre el texto fuente de los agentes y las cadenas
`mcp_servers.*` ya no estarán ahí (viven en `agents/mcp_toolsets.py`).
Reemplazar sus cuerpos:

```python
def test_renewables_uses_grid_data_not_ree_grid():
    src = _agent_source("renewables_forecast")
    assert "GRID_DATA_TOOLSET" in src
    assert "mcp_servers.ree_grid_server" not in src


def test_cross_border_uses_grid_data_not_entsoe():
    src = _agent_source("cross_border")
    assert "GRID_DATA_TOOLSET" in src
    assert "mcp_servers.entsoe_server" not in src
```

```python
def test_regulatory_now_connects_ml_models():
    src = _agent_source("regulatory_compliance")
    assert "ML_TOOLSET" in src
```

Y añadir al final del fichero un test que fije dónde viven las cadenas de servers:

```python
def test_modulos_de_servers_solo_en_mcp_toolsets():
    """Ningun agente define McpToolset propio: las cadenas mcp_servers.* viven
    solo en agents/mcp_toolsets.py."""
    for mod in ("orchestrator", "anomaly_detector", "renewables_forecast",
                "regulatory_compliance", "report_agent", "cross_border"):
        src = _agent_source(mod)
        assert "McpToolset(" not in src, mod
    shared_src = _agent_source("mcp_toolsets")
    for module in ("ml_models_server", "memory_server", "knowledge_base_server",
                   "weather_server", "grid_data_server"):
        assert f"mcp_servers.{module}" in shared_src, module
```

(`test_markov_server_retired_from_all_agents` sigue pasando sin cambios: afirma ausencia.)

- [ ] **Step 10: Suite offline completa**

Run: `cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && PYTHONUTF8=1 python -m pytest -q`
Expected: 248 passed, 16 deselected (246 de Task 1 + test de compartición + test de wiring nuevo)

- [ ] **Step 11: Commit**

```bash
git add agents/ tests/agents/test_shared_toolsets.py tests/agents/test_mcp_wiring.py
git commit -m "refactor(agents): los 6 agentes comparten los 5 toolsets MCP singleton

Elimina las 17 instancias locales de McpToolset (una por agente x server) en
favor de agents/mcp_toolsets.py: maximo 5 subprocesos stdio por corrida.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Documentar el transporte compartido en ARCHITECTURE.md

**Files:**
- Modify: `ARCHITECTURE.md:260` (sección "## 7. Layer 2 — MCP Servers (5 servers)")

**Interfaces:**
- Consumes: nada (solo docs).
- Produces: nota de arquitectura que la demo/lectores del capstone verán.

- [ ] **Step 1: Añadir la nota bajo el encabezado de la sección 7**

Justo después de la línea `## 7. Layer 2 — MCP Servers (5 servers)` (línea 260),
insertar:

```markdown
**Shared stdio transport:** all agents share **a single `McpToolset` instance per
server**, defined in `agents/mcp_toolsets.py`. ADK's `MCPSessionManager` caches the
session (and its subprocess) per toolset instance, so a full orchestrator run spawns
at most 5 subprocesses (one per server) instead of one per agent×server (~17). This
removes the simultaneous-import storm (torch, ~6s cold) that made stdio calls exceed
the 60s timeout during live evals. Design:
`docs/superpowers/specs/2026-07-03-mcp-shared-transport-design.md`.
```

- [ ] **Step 2: Verificación rápida (los tests de docs/wiring no se ven afectados)**

Run: `cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && PYTHONUTF8=1 python -m pytest tests/agents -q`
Expected: todos passed

- [ ] **Step 3: Commit**

```bash
git add ARCHITECTURE.md
git commit -m "docs(arch): transporte stdio compartido — 5 toolsets singleton en la seccion 7

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Validación live y documentación honesta del resultado

**Files:**
- Modify: `eval/README.md:48-62` (sección "Estado de la validación live")
- Modify: `.superpowers/sdd/progress.md` (nueva entrada de sesión)

**Interfaces:**
- Consumes: los agentes migrados (Task 2). Harness: `eval/agents/_harness.py::run_agent_live` (sin cambios).
- Produces: evidencia de si la causa infra quedó eliminada; entrada de ledger para la próxima sesión.

**ATENCIÓN:** esta task consume cuota Gemini real (~7-15 min). Ejecutar la corrida live UNA vez, guardar el log y analizarlo antes de decidir repetir.

- [ ] **Step 1: Línea base offline**

Run: `cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && PYTHONUTF8=1 python -m pytest -q`
Expected: 248 passed, 16 deselected

- [ ] **Step 2: Corrida live completa con log persistido**

Run (un solo comando, timeout 20 min):

```bash
cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && PYTHONUTF8=1 python -m pytest eval/agents -m integration -v 2>&1 | tee eval_live_$(date +%Y%m%d_%H%M).log
```

Expected (criterio de éxito de la spec): `orchestrator/test_critical` y
`renewables/test_load_forecast` PASSED; sin `McpError: Timed out` ni fallback
"ML caído" espurio en el log. Objetivo 12/12; se acepta flakiness LLM residual
(~1 flip) en subagentes SI el log demuestra que ya no hay causa infra.

- [ ] **Step 3: Analizar el log**

Verificar en el log, con grep:

```bash
grep -c "Timed out" eval_live_*.log        # esperado: 0
grep -c "no permitido" eval_live_*.log     # esperado: 0 (guardrail ya corregido)
grep -E "PASSED|FAILED" eval_live_*.log    # tabla de resultados por test
```

Si `orchestrator/test_critical` o `renewables/test_load_forecast` FALLAN con
timeouts MCP: la hipótesis infra era incompleta — NO parchear a ciegas; invocar
superpowers:systematic-debugging con las hipótesis alternativas del handoff
(SQLite lock en gridguard.db, cleanup de event loops de ADK).

- [ ] **Step 4: Actualizar `eval/README.md` con el resultado medido**

Reemplazar el último párrafo de la sección "Estado de la validación live"
(líneas 54-62, desde "Estado por test tras los fixes:") por un párrafo nuevo con
los datos REALES de la corrida del Step 2. Estructura (rellenar con lo medido, no
inventar):

```markdown
Tras compartir los toolsets MCP (spec 2026-07-03, un subproceso por server en vez
de ~17): corrida del 2026-07-0X → N/12 passed. [orchestrator/test_critical: estado
y evidencia] [renewables/test_load_forecast: estado y evidencia] [flips LLM
residuales observados y en qué tests]. [Siguiente paso si queda algo rojo.]
```

- [ ] **Step 5: Actualizar el ledger**

Añadir a `.superpowers/sdd/progress.md` una entrada de sesión 2026-07-03 con: spec
+ plan enlazados, las 4 tasks con su commit hash, resultado de la corrida live
(N/12 y los 2 tests bloqueados), y qué queda para el bloque 2 (flakiness LLM) y
el bloque 3 (cola RAG: 4 temas pending en `data/kb_seed/topics.yaml`).

- [ ] **Step 6: Commit y push**

```bash
git add eval/README.md .superpowers/sdd/progress.md
git commit -m "docs(eval): resultado de la validacion live con toolsets compartidos

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push
```
