# Diseño — Pipeline build-time de ingesta de conocimiento al RAG (2 agentes)

**Fecha:** 2026-07-02
**Rama:** `main`
**Estado:** BORRADOR — pendiente de brainstorming/aprobación del usuario antes de implementar.
**Decisión previa registrada (memoria 2026-07-02):** pipeline **build-time**, NO runtime.

---

## 1. Contexto y problema

El RAG de GridGuard (`knowledge/chroma_db`, colecciones `regulations`/`procedures`/
`incidents`) se puebla hoy con `scripts/seed_knowledge_base.py` a partir de `.txt`
escritos a mano en `data/kb_seed/`. El corpus es pequeño y su crecimiento es manual.
La dirección acordada: dos agentes que enriquecen el conocimiento **offline** (build-time);
el runtime (orchestrator/cross_border) sigue consultando vía `search_regulations` /
`get_mitigation_procedure` sin cambios.

## 2. Agentes propuestos

### 2.1 Agente de investigación (`agents/ingest/researcher` o script ADK offline)

- **Misión:** generar documentos `.txt` de dominio en `data/kb_seed/{regulations,procedures,incidents}/`.
- **Alcance de contenido (del handoff):**
  - POs de REE + normativa ENTSO-E (SO GL, Emergency & Restoration).
  - Procedimientos operativos (deslastre, reposición, reservas).
  - Post-mortems: apagón ibérico 28-abr-2025 y European blackout 4-nov-2006.
  - Acuerdos/relaciones de intercambio con FR/PT/MA (BALIT, MIBEL, ONEE).
- **Patrón probado:** el subagente `general-purpose` que generó
  `data/kb_seed/procedures/interconexiones_red_espanola.txt` (2.027 palabras).
- **Restricciones:**
  - `api-probe` OBLIGATORIO antes de cualquier fetch nuevo (web/APIs).
  - Estilo de los `.txt` existentes: texto plano, secciones en mayúsculas, español.
  - Cada documento incluye sección de fuentes/referencias verificables.
  - Los números operativos (NTCs, umbrales) deben ser consistentes con
    `ml.features.INTERCONNECT_CAPACITY_MW` y el corpus existente (lección de la
    reconciliación ES-PT 1400→3000 / MA HVDC→AC de Task 11).

### 2.2 Agente de embeddings (`scripts/` o `agents/ingest/embedder`)

- **Misión:** trocear + vectorizar + **upsert idempotente** en ChromaDB.
- **Formaliza/reemplaza** `scripts/seed_knowledge_base.py`:
  - Chunking 600/100 (probado).
  - IDs deterministas por (archivo, chunk_idx, hash de contenido) → re-ejecutar no duplica;
    contenido cambiado re-indexa; contenido borrado se poda (reconciliación completa).
  - Modelo SIEMPRE vía `config/model_loader.py::get_model("embedding")` — NUNCA hardcodear.
  - Batch + rate-limit + reintentos (los embeddings Gemini fallan transitoriamente).
- **Shift-left:** el chunking/hash/upsert es Python puro; el único paso "IA" es la llamada
  de embeddings. Evaluar si merece ser "agente" ADK o basta un módulo determinista
  invocado por el pipeline (recomendación preliminar: **módulo determinista**, el agente
  LLM no aporta valor aquí).

## 3. Preguntas abiertas para el brainstorming

1. ¿El agente de investigación usa WebSearch/WebFetch en vivo o solo síntesis del modelo
   con verificación humana? (riesgo de alucinación de números operativos).
2. ¿Revisión humana obligatoria de cada `.txt` antes del seed (gate de calidad), o
   validación automática cruzada contra el corpus existente?
3. ¿Dónde viven: `agents/` (ADK, consistencia con el resto) o `scripts/` (simplicidad,
   no son agentes de runtime)?
4. ¿Se añade un manifiesto (`data/kb_seed/manifest.json`) con hash+fuente+fecha por
   documento para trazabilidad del corpus?

## 4. No-objetivos

- Ingesta en runtime (decidido: NO).
- Cambios en los servidores MCP de consulta (`knowledge_base_server`).
- Scraping masivo de sitios de REE/ENTSO-E.

## 5. Criterios de aceptación (preliminares)

- Re-ejecutar el pipeline completo dos veces deja ChromaDB idéntico (idempotencia).
- `pytest -q` offline verde; smoke tests `integration` de recuperación pasan.
- Ningún nombre de modelo Gemini hardcodeado (auditable por grep).
- Todo documento nuevo pasa la reconciliación de números operativos.
