# Evals de GridGuard — por agente y por eje

Cada agente se evalúa en tres ejes: **trigger** (¿dispara las tools/skills cuando debe y solo cuando debe?), **output** (¿el resultado es correcto?) y **trayectoria** (¿el orden fue el correcto? shift-left Python→ML→LLM).

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

**También por marker:** Para usar markers (pytest.ini define `agent_report`, `agent_orchestrator`, etc.), **el comando correcto incluye `not integration`:**

```bash
python -m pytest -m "agent_report and not integration" -q
```

Nota técnica: En CLI, `-m agent_report` sin restricción de integration ejecutaría los tests `@pytest.mark.integration` asociados al marker, **sobrescribiendo** el `addopts = -m "not integration"` del pytest.ini (CLI gana sobre config). Por eso es obligatorio incluir `and not integration` al usar markers. Por path, en cambio (e.g., `pytest eval/agents/report_agent`), el `addopts` se respeta y los tests integration se deseleccionan automáticamente.

## Advertencia: Tests live/integration

Los tests con `-m integration` requieren **claves de API reales** (GOOGLE_API_KEY, tokens MCP) y consumen cuota LLM. A la fecha (2026-07-02) **no se han ejecutado contra LLM real en CI/CD** — su validación está pendiente. Úsalos solo en entorno local con tokens configurados.

## Flujo prompt→eval

Los prompts del sistema viven en `prompts/<agente>.md`. Tras cambiar un prompt:

1. Edita `prompts/<agente>.md`
2. Corre el eval offline: `python -m pytest eval/agents/<agente> -q`
3. Verifica que los tests pasen
4. Si todos pasan, corre el eval live (si aplica): `python -m pytest eval/agents/<agente> -m integration`
5. Compara resultados antes de commitear

El sistema evalúa tanto **qué tools dispara** (trigger) como **el orden** en que las llamó (trayectoria) y **si el resultado es correcto** (output).

## Extensión futura

Framework nativo de eval de ADK (`.evalset.json` + `agents-cli eval`, LLM-as-judge). No implementado: la suite actual es offline/determinista por diseño (shift-left Python→ML→LLM). Ver spec 2026-07-02.

## Estado de la validación live (primera ejecución: 2026-07-02/03)

Dos corridas completas contra LLM real. Hallazgos corregidos: whitelist del
orchestrator bloqueaba las skills (nombres CamelCase vs runtime snake_case),
timeout MCP de 5s insuficiente (ahora 60s), test de Markov con import obsoleto.

Tras compartir los toolsets MCP (spec 2026-07-03, un subproceso por server en vez
de ~17): corrida del 2026-07-03 → 7/12 passed en 17m08s, con 4 timeouts MCP de 60s
persistentes — la tormenta de subprocesos quedó refutada como causa única.

**Causa raíz encontrada y corregida (2026-07-04, systematic-debugging con py-spy):**
el kb server hacía el import de `chromadb` (que arrastra numpy) de forma perezosa
dentro del primer tool call, con el event loop de anyio ya corriendo; la carga de
la extensión C de numpy se quedaba colgada minutos en `create_module` y el cliente
MCP agotaba su timeout. Repro determinista sin LLM: `search_regulations` vía stdio
pasaba de colgarse >90s a responder en 0.8s moviendo el import al arranque del
módulo (mismo patrón que torch en `ml_models_server`). Fixes acompañantes de la
misma sesión: `timestamp` como str ISO en los 4 output_schema (un `datetime`
rompía `SetModelResponseTool` de ADK 2.3.0 con TypeError), IO externo acotado
(embeddings Gemini con timeout de 30s, ENTSO-E con `timeout=30` y sin retries
largos, forecast 24h con zonas en paralelo), cota dura de 45s por build en
grid-data con caché que ya no guarda errores, y timeout del cliente MCP a 120s.

**Última corrida (2026-07-04, post-fixes): 7/12 passed en 9m41s** (antes 17-27 min)
con **0 timeouts MCP y 0 cotas disparadas** — la causa infra está eliminada.
`renewables/test_load_forecast` PASSED (3 corridas seguidas). Los 5 fallos
restantes son AssertionError de comportamiento del agente, no de infraestructura:
el orchestrator detiene la escalación tras `anomaly_detector_skill` (afecta a
`test_run_scenario_critical`, `test_critical_scenario_triggers_full_escalation` y
al assert de RAG, que solo se satisface si la escalación llega a los sub-agentes),
dispara anomaly en el escenario VERDE (`test_normal_scenario_triggers_nothing`), y
cross_border muestra flips LLM en sus 2 triggers. Próximo paso (decisión de
producto): prompts más directivos para la escalación por capas y/o
`pytest-rerunfailures` para el no-determinismo residual — cada iteración de
validación cuesta ~10 min de cuota Gemini.
