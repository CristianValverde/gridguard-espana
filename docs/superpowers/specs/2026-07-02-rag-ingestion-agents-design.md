# Diseño — Pipeline build-time de ingesta de conocimiento al RAG (2 agentes)

**Fecha:** 2026-07-02
**Rama:** `main`
**Estado:** APROBADO en brainstorming (2026-07-02). Pendiente de plan de implementación.
**Decisiones:** contenido híbrido (síntesis + verificación web de números) · gate
auto + revisión humana · investigador ADK + embedder determinista · manifiesto de
trazabilidad.

---

## 1. Contexto y problema

El RAG de GridGuard (`knowledge/chroma_db`, colecciones `regulations`/`procedures`/
`incidents`) se puebla hoy con `scripts/seed_knowledge_base.py` a partir de `.txt`
escritos a mano en `data/kb_seed/`. El corpus es pequeño, su crecimiento es manual y
ya contenía errores operativos (ES-PT 1400 vs 3000 MW; ES-MA etiquetado HVDC siendo
AC — reconciliados en Task 11 del plan de eficacia). La dirección acordada: pipeline
**build-time** (offline) de dos componentes; el runtime (orchestrator/cross_border)
sigue consultando vía `search_regulations` / `get_mitigation_procedure` sin cambios.

## 2. Flujo general

```
cola de temas → [1] Investigador (ADK, offline) → data/kb_seed/*.txt (status: draft)
             → [2] Validador (Python puro)      → manifest: validated
             → [3] Revisión humana (gate)       → manifest: approved
             → [4] Embedder (Python puro)       → ChromaDB (manifest: indexed)
```

## 3. Componentes

### 3.1 Agente investigador — `agents/ingest/researcher.py` (ADK)

- **Misión:** generar documentos `.txt` de dominio en
  `data/kb_seed/{regulations,procedures,incidents}/` desde una cola de temas
  versionada (`data/kb_seed/topics.yaml`: tema, colección destino, estado):
  POs de REE + normativa ENTSO-E (SO GL, Emergency & Restoration); procedimientos
  operativos (deslastre, reposición, reservas); post-mortems del 28-abr-2025 y del
  4-nov-2006; acuerdos e intercambio con FR/PT/MA (BALIT, MIBEL, ONEE).
- **Modo híbrido:** redacta por síntesis (patrón probado: el doc de interconexiones,
  2.027 palabras), pero **todo número operativo (NTC, umbral, fecha, MW) se verifica
  con búsqueda/fetch web contra fuentes REE/ENTSO-E antes de escribirse**. Si no
  puede verificar un dato, escribe el marcador `[SIN VERIFICAR]` — nunca lo inventa.
- **Estilo:** el de los `.txt` existentes — texto plano, secciones en MAYÚSCULAS,
  español, y sección FUENTES obligatoria con URLs/referencias verificables.
- **Restricciones:** modelo vía `config/model_loader.py::get_model()` (nunca
  hardcodear); `api-probe` OBLIGATORIO antes de integrar cualquier fuente/API nueva.

### 3.2 Validador — `scripts/kb_validate.py` (determinista, sin LLM)

- Consistencia numérica contra las constantes canónicas
  (`ml.features.INTERCONNECT_CAPACITY_MW`) y contra el corpus ya indexado.
- Presencia de sección FUENTES; ausencia de marcadores `[SIN VERIFICAR]` pendientes.
- Si pasa, promueve la entrada del manifiesto a `validated`; si no, informe de fallos.

### 3.3 Manifiesto — `data/kb_seed/manifest.json`

Por documento: `{file, sha256, source_urls, generated_at, reviewed_by, status}`
con `status ∈ {draft, validated, approved, indexed}`. La **aprobación humana**
(paso manual del usuario) es el único gate no automatizado. El propio pipeline
escribe/actualiza el manifiesto.

### 3.4 Embedder — `scripts/kb_embed.py` (determinista, sin LLM)

Formaliza y reemplaza `scripts/seed_knowledge_base.py`:
- Chunking 600/100 (probado).
- IDs deterministas por (archivo, chunk_idx, hash de contenido) → **upsert
  idempotente con poda de huérfanos**: re-ejecutar dos veces deja ChromaDB idéntico;
  contenido cambiado re-indexa; documento retirado se poda.
- Batch + rate-limit + reintentos (los embeddings Gemini fallan transitoriamente).
- Modelo SIEMPRE vía `get_model("embedding")`.
- Solo indexa documentos con `status: approved`.

**Shift-left:** el chunking/hash/upsert es Python puro; el único componente con LLM
es el investigador, donde la redacción/investigación sí lo justifica.

## 4. No-objetivos

- Ingesta en runtime (decidido: NO).
- Cambios en los servidores MCP de consulta (`knowledge_base_server`).
- Scraping masivo de sitios de REE/ENTSO-E (el fetch es puntual, para verificar).

## 5. Manejo de errores

- Embeddings: reintentos con backoff; fallo persistente aborta sin dejar la
  colección a medias (el upsert por documento es atómico respecto al manifest).
- Investigador: dato no verificable → `[SIN VERIFICAR]` (bloquea en el validador,
  no en la generación).
- Validador/embedder: exit code ≠ 0 con informe legible (pipeline scriptable).

## 6. Testing

- **Unit (offline, suite verde):** validador (números inconsistentes → rechaza;
  `[SIN VERIFICAR]` → rechaza); embedder (chunking, IDs estables, idempotencia y
  poda con función de embedding fake y ChromaDB en tmp).
- **Integration (marcados):** recuperación end-to-end tras seed real
  (patrón de `tests/mcp_servers/test_kb_interconnexiones.py`).

## 7. Criterios de aceptación

- Re-ejecutar el pipeline completo dos veces deja ChromaDB idéntico (idempotencia).
- `pytest -q` offline verde; smoke tests `integration` de recuperación pasan.
- Ningún nombre de modelo Gemini hardcodeado (auditable por grep).
- Todo documento nuevo pasa la validación numérica y tiene FUENTES.
- El manifiesto refleja el estado real de cada documento.
