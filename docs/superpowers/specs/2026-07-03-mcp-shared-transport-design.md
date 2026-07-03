# Spec: Toolsets MCP singleton compartidos (transporte compartido)

**Fecha:** 2026-07-03 · **Estado:** aprobada por el usuario
**Contexto previo:** Plan 7 (`2026-07-02-mcp-consolidation-evals-prompts-design.md`) y
sección "Estado de la validación live" de `eval/README.md`.

## Problema

Cada módulo de agente crea sus propias instancias de `McpToolset` stdio: 17 instancias
en 6 módulos para solo 5 servers (`ml_models_server` ×5, `memory_server` ×4,
`knowledge_base_server` ×4, `weather_server` ×2, `grid_data_server` ×2). Un test live
del orchestrator conecta ~17 subprocesos Python simultáneos, cada uno importando su
server a la vez sobre disco OneDrive (el de ML importa torch: ~6s en frío, mucho más
bajo contención). Las llamadas superan el timeout de 60s, el orchestrator activa su
fallback "ML caído" (P=0.20) y la escalación nunca completa. Consecuencia medida:
`orchestrator/test_critical` y `renewables/test_load_forecast` nunca han pasado en live.

Verificado en el código del ADK 2.3.0 instalado (`mcp_session_manager.py`): cada
`McpToolset` tiene su propio `MCPSessionManager` con pool de sesiones cacheado por
(connection params + headers + event loop). El subproceso se lanza una vez por
**instancia** de toolset y se reutiliza entre llamadas del mismo loop. El problema no
es el transporte stdio en sí: es que hay 17 instancias donde bastan 5.

## Decisión

**Compartir una única instancia de `McpToolset` por server entre todos los agentes**
(evals y runtime real). Alternativas descartadas:

- **HTTP streamable/SSE con servers arrancados una vez:** más robusto a largo plazo,
  pero exige lifecycle manager (puertos, salud, parada) en Windows, cambios en los
  5 servers y protocolo `api-probe` previo — demasiadas piezas a 3 días del deadline
  del capstone (06/07). Queda como **evolución futura** (ver abajo).
- **Precalentamiento (fixture pytest de sesión):** no reduce los subprocesos, solo
  templa la caché de disco; no ataca la causa raíz y no sirve al runtime.

## Diseño

### 1. Módulo nuevo: `agents/mcp_toolsets.py`

Define las 5 instancias compartidas mediante un helper interno:

```python
def _stdio_toolset(module: str) -> McpToolset:
    return McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command="python", args=["-m", module],
            ),
            timeout=60.0,
        )
    )

ML_TOOLSET        = _stdio_toolset("mcp_servers.ml_models_server")
MEMORY_TOOLSET    = _stdio_toolset("mcp_servers.memory_server")
KB_TOOLSET        = _stdio_toolset("mcp_servers.knowledge_base_server")
WEATHER_TOOLSET   = _stdio_toolset("mcp_servers.weather_server")
GRID_DATA_TOOLSET = _stdio_toolset("mcp_servers.grid_data_server")
```

### 2. Migración de los 6 módulos de agentes

`orchestrator.py`, `anomaly_detector.py`, `renewables_forecast.py`,
`regulatory_compliance.py`, `report_agent.py` y `cross_border.py` eliminan sus
definiciones locales `_*_toolset = McpToolset(...)` e importan las compartidas.
Las listas `tools=[]` referencian las mismas capas en el mismo orden. **Nada más
cambia:** patrón `as_static_instruction` + `load_prompt`, guardrails
(`register_hooks`), prompts y sub_agents quedan intactos.

### 3. Comportamiento en runtime

- El primer agente que llama a un tool de un server paga el arranque (import de
  torch una sola vez); el resto reutiliza la sesión cacheada del session manager
  compartido.
- Dentro de una corrida del orchestrator todo ocurre en el mismo event loop →
  **máximo 5 subprocesos** (antes ~17).
- Entre tests de pytest (event loops distintos) el ADK detecta el cambio de loop y
  recrea la sesión: respawn secuencial por server, no tormenta, con caché de disco
  ya caliente.
- Concurrencia: las skills se invocan secuencialmente como AgentTools en el flujo
  actual; en caso de llamadas paralelas, `ClientSession` de MCP multiplexa por
  JSON-RPC id y `create_session` está protegido por lock (verificado en ADK).

### 4. Manejo de errores

Sin código nuevo: el `MCPSessionManager` del ADK ya detecta sesión desconectada o
loop muerto y recrea (`mcp_session_manager.py:734-739`). Timeout se mantiene en 60s.
Los warnings "original event loop is closed" al cierre del proceso pueden persistir:
son cosméticos (cleanup best-effort del ADK) y quedan anotados como conocidos.

### 5. Testing

1. **Suite offline intacta (243 passed):** la creación de toolsets a nivel de módulo
   no lanza subprocesos (spawn lazy en la primera llamada), igual que hoy.
2. **Test unitario nuevo de compartición:** recorre las `tools` de todos los agentes
   (incluidos los inner agents de las Skills) y afirma que el conjunto de instancias
   `McpToolset` únicas es exactamente el de las 5 compartidas — evita regresiones
   por redefiniciones locales.
3. **Validación live (bloque 2 del handoff):** re-correr
   `PYTHONUTF8=1 python -m pytest eval/agents -m integration` con foco en
   `orchestrator/test_critical` y `renewables/test_load_forecast`. Solo tras
   eliminar esta causa infra se valorará mitigación de flakiness LLM residual.

### 6. Documentación

- `ARCHITECTURE.md` §7: nota de que los agentes comparten 5 toolsets singleton
  (un subproceso por server, no por agente×server).
- `eval/README.md`, sección "Estado de la validación live": actualizar tras la
  corrida con resultados honestos.

## Evolución futura (no implementar ahora)

Transporte HTTP streamable de FastMCP con los 5 servers como servicios arrancados
una vez por sesión de trabajo (`StreamableHTTPConnectionParams` en los agentes).
Eliminaría también el respawn entre event loops de pytest. Requiere `api-probe`
del transporte antes de integrar y un gestor de ciclo de vida de procesos.

## Criterio de éxito

- Suite offline verde (243+ tests, con el nuevo test de compartición).
- `orchestrator/test_critical` y `renewables/test_load_forecast` en verde en live
  (objetivo del handoff: 12/12; mínimo: los 2 bloqueados por infra).
- Sin fallback "ML caído" espurio ni `McpError: Timed out` en los logs live.
