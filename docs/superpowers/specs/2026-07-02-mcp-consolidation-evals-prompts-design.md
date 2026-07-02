# Spec — Consolidación MCP (7→5) + evals por agente + prompts versionables

**Fecha:** 2026-07-02 · **Estado:** aprobada por el usuario (brainstorming de esta sesión)
**Decisiones del usuario:** consolidación media (7→5) · poller solo lectura · evals con pytest por agente.

## Contexto y evidencia (auditoría de uso real)

Mapa verificado en código (`guardrails/hooks.py::_ALLOWED_TOOLS` × `McpToolset` en `agents/*` × `@mcp.tool()` en `mcp_servers/*`):

| Server | Tools | Conectado por | Tools huérfanas (sin consumidor) |
|---|---|---|---|
| `ree-grid` | 5 | solo renewables_forecast | `get_realtime_demand`, `get_interchange_flows`, `get_programmed_generation`, `get_reserve_activation` (4/5) |
| `weather` | 4 | anomaly, renewables | `get_aemet_alerts` |
| `entso-e` | 4 | solo cross_border | ninguna |
| `memory` | 7 | orch, anomaly, renewables, report | ninguna |
| `ml-models` | 8 | orch, anomaly, renewables, report | ninguna (`reload_models` es operacional: `reload_mcp_server.py`) |
| `knowledge-base` | 3 | anomaly, report, cross_border, regulatory | ninguna |
| `markov` | 3 | 5 de 6 agentes | `get_transition_probabilities` |

Bugs confirmados:
- **IDs ESIOS erróneos en dos sitios**: `mcp_servers/ree_grid_server.py::_INDICATORS` y `tools/ree_poller.py` usan nuclear=460, hydro=1163, wind=551, solar_thermal=1160. Los verificados (en `ml/data/download_esios.py`) son nuclear=1153, hydro=10035, wind=10037, solar_thermal=1162 (solar_pv=1161 sí coincide). El comentario "mismos IDs que download_esios.py" en ree_grid_server es falso.
- **Doble escritor potencial de `snapshots`**: `incremental_load.catch_up` escribe a escala de entrenamiento (la que consume `get_serve_features`); el flujo orchestrator→`store_snapshot` escribiría escala cruda del poller, contaminando la tabla.

## Sección 1 — Consolidación de MCP servers (7 → 5)

Resultado: `grid-data`, `weather`, `memory`, `ml-models`, `knowledge-base`.

1. **Nuevo `mcp_servers/grid_data_server.py`** (MCP `gridguard-grid-data`):
   - Absorbe las 4 tools de `entsoe_server.py` sin cambios: `get_cross_border_flows`, `get_eu_generation`, `get_capacity_margin`, `get_eu_market_prices`.
   - Absorbe de `ree_grid_server.py` **solo** `get_generation_mix`, con IDs verificados.
   - Credenciales: `ESIOS_TOKEN` + `ENTSOE_TOKEN` (cada tool usa la suya; fallo de una no afecta a la otra).
   - Se retiran `entsoe_server.py` y `ree_grid_server.py` (con sus 4 tools huérfanas).
2. **Fold de `markov_server.py` en `ml_models_server.py`**:
   - Se mueven `get_grid_state` y `get_markov_trajectory` (delegan en `tools/markov_state_tracker`; determinista, sin credenciales).
   - `get_transition_probabilities` se elimina (huérfana).
   - Los nombres de tools no cambian → `_ALLOWED_TOOLS` no cambia por este motivo.
   - Colateral aceptado: `regulatory_compliance` pasa a conectar `ml-models` (carga `.pkl` al arranque; coste menor asumido).
3. **`weather_server.py`**: eliminar `get_aemet_alerts`. El server permanece.
4. **Poller solo lectura**:
   - Corregir IDs en `tools/ree_poller.py`.
   - Quitar `store_snapshot` de la whitelist del orchestrator y del paso correspondiente de su instruction.
   - `snapshots` queda con un único escritor: `incremental_load.catch_up`.
   - `store_snapshot` se elimina de `memory_server` si la auditoría de implementación confirma que no queda ningún consumidor (scripts incluidos); si queda alguno, se conserva y solo se retira del orchestrator.
5. **Conexiones de agentes**: renewables_forecast y cross_border → `grid-data`; las conexiones a markov se sustituyen por `ml-models` (ya presente en 4 de los 5; regulatory la añade).

## Sección 2 — Evals por agente (pytest, offline)

Reestructurar `eval/` por agente en vez de por tipo:

```
eval/
  agents/
    orchestrator/            # test_trajectory.py, test_outputs.py
    anomaly_detector/
    renewables_forecast/
    regulatory_compliance/
    report_agent/
    cross_border/
  ml/                        # test_ml_models.py (capa ML, transversal)
  golden_dataset/            # sin cambios
```

- Los tests actuales de `test_trajectories.py` / `test_outputs.py` se **reparten** entre carpetas, no se reescriben.
- Invocación individual por path (`python -m pytest eval/agents/report_agent -q`) y por marker (`-m agent_report`), markers registrados en `pytest.ini`.
- Comando por agente documentado (README de `eval/` o sección en ARCHITECTURE.md).
- El framework nativo de ADK (`.evalset.json` + `agents-cli eval`, LLM-as-judge) queda como **extensión futura documentada**, no se implementa.

## Sección 3 — Prompts versionables

- Extraer los `_INSTRUCTION` de los 6 agentes a `prompts/<agente>.md` (un fichero por agente).
- Carga en import mediante helper (en `agents/instruction_utils.py` o `prompts/loader.py`) que lee el fichero y lo pasa por `as_static_instruction`. **El patrón `as_static_instruction` no cambia** (evita el KeyError de templating de ADK 2.3.0 con llaves literales).
- Test mínimo: cada agente carga un prompt no vacío desde su fichero.
- Sinergia con la sección 2: cambio de prompt → correr el eval de ese agente y comparar.

## Orden de implementación

Estricto **1 → 2 → 3**: no estructurar carpetas de evals/prompts alrededor de servers que se fusionan o retiran.

## Verificación (criterios de cierre)

- `python -m pytest -q` verde (205+ actuales; los tests de servers retirados se migran al server nuevo, sin perder capacidades testeadas).
- Whitelists de `guardrails/hooks.py` coherentes con los 5 servers (único cambio real: `store_snapshot` fuera del orchestrator).
- Evals invocables por agente, comando documentado.
- Docs actualizadas: `ARCHITECTURE.md` §7 (5 servers), §13 (árbol), mención "7 custom MCP servers" (línea ~627) y `architecture_diagram.html`; `.superpowers/sdd/progress.md`; memoria del proyecto.
- Commits en español (conventional) sobre `main` + push a origin.

## Fuera de alcance

- Framework de eval nativo de ADK (documentado como futuro).
- LangChain/LangGraph (descartado pre-capstone; decisión cerrada).
- Backlog oportunista del Plan 6 (matching por proximidad en `check_capacity_consistency`, etc.) — solo si cae de forma natural durante la implementación.
