# Lessons Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dar al orchestrator memoria de "lecciones aprendidas" entre ciclos de monitorización (sesiones efímeras) mediante una tabla SQLite nueva y dos tools MCP (`store_lesson`, `get_lessons`) en el memory server.

**Architecture:** Tabla `lessons` en `gridguard.db` (mismo patrón que `risk_scores`/`grid_states`). El prompt del orchestrator consulta lecciones relevantes al inicio del ciclo y registra una nueva al cierre SOLO en ciclos relevantes (NARANJA/ROJO, fallo de tool, o predicción contradicha). SQLite, no ChromaDB — decisión razonada en la spec (volumen bajo, sin coste de embeddings, testeable offline).

**Tech Stack:** sqlite3 (stdlib), pydantic (`schemas/models.py`), FastMCP (`mcp_servers/memory_server.py`), mismo patrón de tests con DB temporal (`tmp_path`) que `tests/mcp_servers/test_memory_grid_states.py`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-04-lessons-memory-design.md`. Depende de que el plan `docs/superpowers/plans/2026-07-04-escalation-planner.md` haya llegado (al menos) a su Task 3 antes de ejecutar la Task 4 de este plan (ambos tocan `prompts/orchestrator.md`) — ver nota en esa Task.
- Trabajar sobre `main` del proyecto `"C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project"`. Prefijar CADA comando con `cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && `.
- `PYTHONUTF8=1` en todos los comandos python.
- NUNCA leer `.env`, `*.key`, `secrets.*`, `credentials.*`.
- NUNCA hardcodear modelos Gemini.
- `timestamp` en el schema `Lesson` es `str` ISO 8601 (NO `datetime`) — convención post-fix `SetModelResponseTool` de ADK 2.3.0: un `datetime` en un output_schema rompe la serialización con TypeError. Usar el helper `_utcnow_iso` ya definido en `schemas/models.py`.
- `python -m pytest -q` corre solo la suite offline. NO ejecutar tests `-m integration` en este plan salvo en la Task 4 (validación live COMPARTIDA con el plan del planner — solo dispararla cuando ambos planes estén completos).
- Commits en español, conventional, con trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Solo el orchestrator escribe lecciones en esta versión (v1); no tocar subagentes.

---

### Task 1: Schema `Lesson` en `schemas/models.py`

**Files:**
- Modify: `schemas/models.py`
- Test: `tests/agents/test_output_schema_json_safe.py` (añadir `Lesson` a la lista de instancias verificadas — reutiliza el patrón existente)

**Interfaces:**
- Produces: `class Lesson(BaseModel)` con campos `lesson: str`, `context: str`, `markov_state: int` (`ge=0, le=4`), `p_band: Literal["VERDE","AMARILLO","NARANJA","ROJO"]`, `incident_type: Optional[str] = None`, `incident_id: Optional[str] = None`, `source_agent: str = "gridguard_orchestrator"`, `timestamp: str` (default `_utcnow_iso`). Consumido por Task 2 (memory server) y Task 4 (prompt).

- [ ] **Step 1: Leer el patrón existente y localizar el punto de inserción**

Run: `cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && grep -n "^class CrossBorderAnalysis" schemas/models.py`

`Lesson` se añade DESPUÉS de la clase `CrossBorderAnalysis` (fin de la sección
"Capa 4 — Sub-agent outputs") y ANTES de la sección "Cross-cutting — Guardrails / Hooks"
(marcador `# Cross-cutting`).

- [ ] **Step 2: Escribir el test que falla**

Añadir a `tests/agents/test_output_schema_json_safe.py`, en el import y en `_instances()`:

```python
from schemas.models import (
    AnomalyReport,
    ComplianceReport,
    CrossBorderAnalysis,
    Lesson,
    RenewablesForecast,
)
```

```python
def _instances():
    return [
        AnomalyReport(
            anomalies=[], severity="LOW", confidence=0.5, ml_score=0.1,
            regulatory_context="", reasoning="x",
        ),
        RenewablesForecast(
            risk_level="LOW", expected_generation_pct=40.0,
            critical_window="14:00-18:00 UTC", ml_forecast_mw=25000.0,
            reasoning="x",
        ),
        ComplianceReport(compliance_status="compliant", reasoning="x"),
        CrossBorderAnalysis(
            can_import=True, available_capacity_mw=1000.0,
            countries=["FR"], risk_mitigation="x",
        ),
        Lesson(
            lesson="Escalar a AMARILLO aunque P<0.10 si Markov ya está en S2.",
            context="Ciclo 2026-07-04T10:00Z: P=0.08 pero Markov=S2 tras rampa solar.",
            markov_state=2, p_band="AMARILLO",
        ),
    ]
```

Run: `cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && PYTHONUTF8=1 python -m pytest tests/agents/test_output_schema_json_safe.py -v`
Expected: ERROR `cannot import name 'Lesson' from 'schemas.models'`

- [ ] **Step 3: Implementar `Lesson`**

En `schemas/models.py`, insertar tras `CrossBorderAnalysis` (antes del comentario
`# Cross-cutting — Guardrails / Hooks`):

```python
class Lesson(BaseModel):
    """Lección aprendida por el orquestador al cierre de un ciclo relevante.

    Memoria entre sesiones (cada ciclo del orchestrator es efímero): permite
    consultar decisiones pasadas relevantes al arrancar el siguiente ciclo.
    """

    lesson: str = Field(..., description="La lección, imperativa y accionable")
    context: str = Field(..., description="Resumen de la situación que la produjo")
    markov_state: int = Field(..., ge=0, le=4)
    p_band: Literal["VERDE", "AMARILLO", "NARANJA", "ROJO"]
    incident_type: Optional[str] = Field(
        None, description="Alineado con AnomalyReport.incident_type"
    )
    incident_id: Optional[str] = Field(None, description="FK lógica a incidents.incident_id")
    source_agent: str = "gridguard_orchestrator"
    timestamp: str = Field(default_factory=_utcnow_iso, description="ISO 8601 UTC")
```

- [ ] **Step 4: Verificar que pasa**

Run: `cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && PYTHONUTF8=1 python -m pytest tests/agents/test_output_schema_json_safe.py -v`
Expected: passed (ahora 5 instancias en `_instances()`, ambos tests parametrizados cubren `Lesson`)

- [ ] **Step 5: Suite offline completa**

Run: `cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && PYTHONUTF8=1 python -m pytest -q`
Expected: 261 passed, 16 deselected (sin nuevos test_ funcs — se ampliaron los parametrizados existentes; si el proyecto base ya tiene 278 por el plan del planner ejecutado antes, ajustar el número base en consecuencia y verificar que solo sube por los tests que SÍ son nuevos en este plan)

- [ ] **Step 6: Commit**

```bash
git add schemas/models.py tests/agents/test_output_schema_json_safe.py
git commit -m "feat(schemas): Lesson — memoria de lecciones aprendidas entre ciclos del orchestrator

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Tabla `lessons` + tools `store_lesson`/`get_lessons` en el memory server

**Files:**
- Modify: `mcp_servers/memory_server.py`
- Test: `tests/mcp_servers/test_memory_lessons.py`

**Interfaces:**
- Consumes: `Lesson` de `schemas/models.py` (Task 1).
- Produces: tool MCP `store_lesson(lesson_json: str) -> dict` (`{"ok": True, "id": int}` o `{"error": str}`); tool MCP `get_lessons(markov_state: int = -1, p_band: str = "", limit: int = 10) -> list[dict]`. Consumido por Task 3 (whitelist) y Task 4 (prompt).

- [ ] **Step 1: Escribir los tests que fallan**

Crear `tests/mcp_servers/test_memory_lessons.py` (mismo patrón que
`tests/mcp_servers/test_memory_grid_states.py`: DB temporal vía `tmp_path`,
funciones `_impl` inyectables con `db_path`):

```python
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
```

- [ ] **Step 2: Verificar que fallan**

Run: `cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && PYTHONUTF8=1 python -m pytest tests/mcp_servers/test_memory_lessons.py -v`
Expected: todos ERROR con `AttributeError: module 'mcp_servers.memory_server' has no attribute '_init_db_at'` o `'_store_lesson_impl'` (verificar primero si `_init_db_at` ya existe — está referenciado en `test_memory_grid_states.py:6`; si ya existe, el error será solo por `_store_lesson_impl`/`_get_lessons_impl`)

- [ ] **Step 3: Añadir la tabla al schema**

En `mcp_servers/memory_server.py`, dentro de `_init_db()` (el `executescript` que
crea `snapshots`/`risk_scores`/`incidents`/`grid_states`), añadir tras la última
tabla (`grid_states`) y su índice:

```python
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
```

Si existe también un `_init_db_at(db_path)` separado (usado por
`test_memory_grid_states.py`), replicar el mismo bloque `CREATE TABLE
lessons`/`CREATE INDEX` ahí (buscar con
`grep -n "_init_db_at\|_init_db_at" mcp_servers/memory_server.py` para confirmar
si hay una única función que ambos tests reutilizan, o dos separadas).

- [ ] **Step 4: Implementar `_store_lesson_impl` y `_get_lessons_impl`**

Añadir el import de `Lesson` al bloque de imports existente:

```python
from schemas.models import Baseline, Incident, Lesson, RiskScore
```

Añadir, cerca de las funciones `_impl` de `grid_states` (mismo estilo:
funciones puras con `db_path` inyectable, conexión propia con
`sqlite3.connect`):

```python
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
```

- [ ] **Step 5: Verificar que pasan**

Run: `cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && PYTHONUTF8=1 python -m pytest tests/mcp_servers/test_memory_lessons.py -v`
Expected: 8 passed

- [ ] **Step 6: Suite offline completa**

Run: `cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && PYTHONUTF8=1 python -m pytest -q`
Expected: +8 sobre la línea base de Task 1 de este plan

- [ ] **Step 7: Commit**

```bash
git add mcp_servers/memory_server.py tests/mcp_servers/test_memory_lessons.py
git commit -m "feat(memory): tabla lessons + store_lesson/get_lessons (memoria entre ciclos)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Whitelist y ARCHITECTURE.md

**Files:**
- Modify: `guardrails/hooks.py:108-116` (bloque `_ALLOWED_TOOLS["gridguard_orchestrator"]`)
- Modify: `eval/agents/orchestrator/test_trigger.py:13-21` (`_EXPECTED_TOOLS`)
- Modify: `ARCHITECTURE.md` (sección `### MCP: memory`, §7)

**Interfaces:**
- Consumes: nombres de tool `store_lesson`, `get_lessons` (Task 2).
- Produces: whitelist y test de wiring actualizados; documentación sincronizada.

- [ ] **Step 1: Añadir las 2 tools a la whitelist**

En `guardrails/hooks.py`, dentro de `_ALLOWED_TOOLS["gridguard_orchestrator"]`,
añadir `"store_lesson"` y `"get_lessons"` al set. Si el plan del planner
(`2026-07-04-escalation-planner.md`) ya se ejecutó antes, el set ya contiene
`"escalation_planner"` — añadir las 2 entradas nuevas sin tocar las existentes:

```python
    "gridguard_orchestrator": {
        "ree_poller", "data_normalizer", "stability_proxy_calculator",
        "risk_scorer", "escalation_planner", "alert_formatter",
        "get_blackout_probability", "store_risk_score",
        "get_serve_features", "get_ensemble_score", "get_blackout_probability_ensemble",
        "anomaly_detector_skill", "renewables_forecast_skill",
        "cross_border_agent", "report_agent",
        "get_grid_state", "store_grid_state", "regulatory_compliance_skill",
        "store_lesson", "get_lessons",
    },
```

(Si `escalation_planner` NO está presente porque este plan se ejecuta antes que
aquel, omitir esa entrada del bloque anterior — añadir solo `store_lesson` y
`get_lessons` al set existente tal como esté en ese momento.)

- [ ] **Step 2: Actualizar el test de wiring del orchestrator**

En `eval/agents/orchestrator/test_trigger.py`, añadir `"store_lesson"` y
`"get_lessons"` a `_EXPECTED_TOOLS` (mismo set que la whitelist del Step 1,
verbatim).

- [ ] **Step 3: Actualizar ARCHITECTURE.md**

En la sección `### MCP: memory` (§7), añadir a la lista de tools expuestas:

```
store_lesson(lesson_json) -> dict          # persist a learned lesson
get_lessons(markov_state, p_band, limit) -> list[dict]  # retrieve past lessons
```

- [ ] **Step 4: Suite offline completa**

Run: `cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && PYTHONUTF8=1 python -m pytest -q`
Expected: mismo número que el final de Task 2 (solo wiring, sin tests nuevos)

- [ ] **Step 5: Commit**

```bash
git add guardrails/hooks.py eval/agents/orchestrator/test_trigger.py ARCHITECTURE.md
git commit -m "feat(orchestrator): whitelist + docs para store_lesson/get_lessons

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Wiring del prompt (consulta + registro de lecciones)

**Files:**
- Modify: `prompts/orchestrator.md`

**Interfaces:**
- Consumes: `get_lessons`/`store_lesson` (Task 2-3); si el plan del planner ya corrió,
  también el PASO 2 reescrito de ese plan (esta task añade un sub-paso al PASO 1
  y una sección PASO 3 nueva, sin tocar el PASO 2).
- Produces: prompt final que el LLM sigue en runtime — validado en la corrida
  live compartida (Task 5 del plan del planner).

**IMPORTANTE — orden de ejecución:** si el plan `2026-07-04-escalation-planner.md`
todavía no llegó a su Task 3 (reescritura del PASO 2), ejecutar esa Task 3
primero y DESPUÉS esta Task 4, para evitar conflictos de merge manual en el
mismo fichero. Si ya se ejecutó, este Step edita el prompt ya actualizado por
aquel plan.

- [ ] **Step 1: Añadir el sub-paso de consulta al PASO 1**

En `prompts/orchestrator.md`, en el PASO 1 (adquisición de datos), añadir un
sub-paso nuevo justo después de `1a` (`ree_poller`) — el orden exacto de
inserción depende de si el PASO 1h del plan del planner ya existe; en cualquier
caso, este sub-paso de lecciones va ANTES del sub-paso de escalación (si existe)
porque debe informar la decisión, no seguirla:

```
  1a-bis. Llama a `get_lessons` con markov_state=<state de get_grid_state, o -1
      si aun no lo has calculado en este punto> y p_band="" (sin filtro de banda
      en esta primera consulta). TEN EN CUENTA las lecciones devueltas al decidir
      la escalacion mas adelante en este ciclo. Si alguna lección condicionó tu
      decisión, cítala en `actions_taken` (p.ej. "aplicada lección: <texto>").
```

(Si el PASO 1 ya tiene sub-pasos etiquetados `1a`-`1h` por el plan del planner,
usar la etiqueta `1i` en vez de `1a-bis` y colocarla al final del PASO 1, ya que
para tener el `markov_state` real (del PASO 1d) es más simple consultar
lecciones después de tener el estado. Ajustar el texto de arriba a: "Llama a
`get_lessons` con markov_state=<state del paso 1d>".)

- [ ] **Step 2: Añadir la sección PASO 3 — Aprendizaje**

Al final del fichero, después del bloque `[ROJO -- EMERGENCIA]` (o después del
PASO 2 reescrito si el plan del planner ya corrió) y ANTES de la sección
"REGLAS NO NEGOCIABLES", añadir:

```
================================================
PASO 3 -- APRENDIZAJE (al cierre del ciclo)
================================================
Registra UNA leccion con `store_lesson` SOLO si el ciclo fue relevante:
  - La banda de escalacion fue NARANJA o ROJO, O
  - Alguna tool critica fallo (ree_poller, get_blackout_probability_ensemble,
    risk_scorer), O
  - El resultado contradijo la prediccion (ej. Markov escalo pese a P baja,
    o P fue alta pero Markov se mantuvo estable).

La leccion debe ser CORTA, imperativa y accionable (ej. "Escalar a AMARILLO
aunque P<0.10 si Markov ya esta en S2"), NO un resumen del ciclo. Incluye:
  - lesson: la leccion en si
  - context: 1-2 frases de la situacion que la produjo
  - markov_state: el estado Markov de este ciclo
  - p_band: la banda de escalacion de este ciclo

En ciclos VERDES rutinarios (sin ninguna de las condiciones anteriores) NO
registres ninguna leccion -- evita ruido en la memoria.
```

- [ ] **Step 3: Verificar que el prompt sigue cargando sin error de templating**

Run: `cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && PYTHONUTF8=1 python -m pytest tests/agents/test_prompt_files.py tests/agents/test_instruction_templating.py -v`
Expected: todos passed

- [ ] **Step 4: Suite offline completa**

Run: `cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && PYTHONUTF8=1 python -m pytest -q`
Expected: mismo número que el final de Task 3

- [ ] **Step 5: Commit**

```bash
git add prompts/orchestrator.md
git commit -m "feat(orchestrator): PASO 1 consulta lecciones, PASO 3 registra al cierre del ciclo

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

**Nota final:** la validación live de este plan está cubierta por la Task 5 del
plan `2026-07-04-escalation-planner.md` (corrida compartida) — no repetir aquí.
