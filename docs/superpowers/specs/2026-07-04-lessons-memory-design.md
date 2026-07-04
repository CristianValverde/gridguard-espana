# Spec: memoria de lecciones aprendidas del orquestador (SQLite)

**Fecha:** 2026-07-04 · **Estado:** aprobada por el usuario (alternativa 1)
**Contexto:** cada sesión del orquestador es un ciclo de monitorización efímero
(`InMemorySessionService` local a `run()`). Entre ciclos sobrevive estado
operativo (risk_scores, grid_states, incidents en `gridguard.db`) pero NO hay
lecciones aprendidas: nada registra "qué salió mal/bien y qué hacer distinto" ni
nada lo consulta al arrancar el siguiente ciclo.

## Decisión: SQLite en el memory server, NO ChromaDB

Razonada con el usuario (2026-07-04):
1. **Volumen**: decenas de lecciones, no miles — con filtros estructurados
   (estado Markov, banda, tipo de incidente) el agente puede leerse las top-N
   directamente; la búsqueda semántica es overkill a este volumen.
2. **Coste/fiabilidad**: Chroma exigiría un embedding Gemini por escritura y
   consulta (cuota compartida con los agentes, y venimos de arreglar cuelgues en
   esa ruta). SQLite es local, determinista y testeable offline (shift-left).
3. **Infraestructura existente**: `gridguard.db` ya tiene `incidents` para
   enlazar; el memory server ya es la memoria entre sesiones del orquestador.
4. **Puerta abierta**: si crecen a cientos, el pipeline de ingesta existente
   puede embeberlas a Chroma después; SQLite queda como fuente de verdad.

## Diseño

### 1. Schema pydantic `Lesson` (`schemas/models.py`)

```python
class Lesson(BaseModel):
    """Lección aprendida por el orquestador al cierre de un ciclo relevante."""
    lesson: str                      # la lección, imperativa y accionable
    context: str                     # resumen de la situación que la produjo
    markov_state: int = Field(..., ge=0, le=4)
    p_band: Literal["VERDE", "AMARILLO", "NARANJA", "ROJO"]
    incident_type: Optional[str] = None   # alineado con AnomalyReport.incident_type
    incident_id: Optional[str] = None     # FK lógica a incidents.incident_id
    source_agent: str = "gridguard_orchestrator"
    timestamp: str = Field(default_factory=_utcnow_iso, description="ISO 8601 UTC")
```

`timestamp` es `str` ISO por la convención post-fix de `SetModelResponseTool`
(los datetime rompen la serialización de ADK 2.3.0).

### 2. Tabla y tools en `mcp_servers/memory_server.py`

Tabla nueva en `_init_db()` (mismo patrón `CREATE TABLE IF NOT EXISTS`):

```sql
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

Tools MCP (síncronas, locales, mismo estilo que store_risk_score):
- `store_lesson(lesson_json: str) -> dict` — valida con `Lesson.model_validate_json`,
  inserta, devuelve `{"ok": True, "id": ...}`. Errores de validación →
  `{"error": ...}` (no excepción).
- `get_lessons(markov_state: int = -1, p_band: str = "", limit: int = 10) -> list[dict]`
  — filtros opcionales (−1/"" = sin filtro), orden `timestamp DESC`, `limit`
  recortado a [1, 50].

### 3. Prompt `prompts/orchestrator.md`

- PASO 1 gana un sub-paso inicial: llama a `get_lessons` con el estado Markov
  actual (y sin filtro de banda) y TEN EN CUENTA las lecciones devueltas al
  decidir; cítalas en `actions_taken` si condicionaron una decisión.
- Sección nueva "PASO 3 — APRENDIZAJE (al cierre del ciclo)": registra UNA
  lección con `store_lesson` SOLO si el ciclo fue relevante: banda NARANJA/ROJO,
  o una tool crítica falló, o el resultado contradijo la predicción (p.ej.
  Markov escaló pese a P baja). La lección debe ser corta, imperativa y
  accionable; en ciclos VERDES rutinarios NO se registra nada (evitar ruido).

### 4. Wiring

- `guardrails/hooks.py`: añadir `"store_lesson"` y `"get_lessons"` a
  `_ALLOWED_TOOLS["gridguard_orchestrator"]`.
- El orquestador ya tiene `MEMORY_TOOLSET`: las tools nuevas llegan solas por el
  server; no cambia `agents/orchestrator.py`.
- `ARCHITECTURE.md` §7 (`MCP: memory`): añadir las 2 tools a la lista.

### 5. Testing

1. Unit tests del memory server con DB temporal (patrón `_init_db_at`/impl de
   `store_grid_state`): insertar + recuperar, filtros por estado/banda, orden
   DESC, limit recortado, JSON inválido → `{"error": ...}` sin excepción.
2. Schema: `Lesson` serializa con `json.dumps(model_dump(exclude_none=True))`
   (añadirlo a `tests/agents/test_output_schema_json_safe.py` si se reutiliza el
   patrón, o test propio).
3. Wiring: whitelist contiene las 2 tools; el prompt referencia `get_lessons` y
   `store_lesson` (asserts de texto).
4. Live: cubierto por la corrida única compartida con la spec del planner —
   verificar en el log que un escenario ROJO produce un `store_lesson` en la
   trayectoria (assert blando: si el LLM no lo llama, documentarlo, no bloquear
   el criterio de éxito del planner).

## Criterio de éxito

- Suite offline verde con los tests nuevos.
- `store_lesson`/`get_lessons` funcionales vía MCP (cubiertos por unit tests
  offline; humo live opcional en la corrida compartida).
- El prompt integra consulta y registro sin romper ningún eval offline existente.

## Fuera de alcance (post-capstone)

- Embedding de lecciones a Chroma vía pipeline de ingesta.
- Curación/deduplicación automática de lecciones; TTL o poda.
- Lecciones escritas por subagentes (solo el orquestador escribe en v1).
