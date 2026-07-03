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
de ~17): corrida del 2026-07-03 → **7/12 passed** (5 failed, 1028.90s / 17m08s).
`renewables/test_load_forecast` (`test_consults_weather_and_load_forecast`)
**PASSED**. `orchestrator/test_critical` (`test_run_scenario_critical`) **FAILED**:
el log muestra `McpError: Timed out while waiting for response to ClientRequest.
Waited 60.0 seconds.` dentro del mismo bloque de captura del test — la tormenta de
subprocesos ya no es la causa (solo hay 1 proceso por server ahora), pero el
timeout MCP de 60s sigue apareciendo (4 ocurrencias en el log completo, repartidas
entre `test_ample_margin_scenario_skips_mitigation`, `test_run_scenario_critical`,
`test_critical_scenario_triggers_full_escalation` y
`test_normal_scenario_triggers_nothing`, todos del orchestrator salvo el primero).
Sin fallback "ML caído" espurio (0 ocurrencias) y sin bloqueos del guardrail
("no permitido": 0 ocurrencias). De los 5 fallos, 4 tienen un timeout MCP de 60s
dentro de su propio bloque de captura (`test_ample_margin_scenario_skips_mitigation`,
`test_run_scenario_critical`, `test_critical_scenario_triggers_full_escalation`,
`test_normal_scenario_triggers_nothing`); el quinto
(`test_consults_ml_and_baseline`, subagente anomaly_detector) falla por assertion
pura sin timeout visible en su bloque — no-determinismo LLM (nombre de tool
distinto al esperado, ej. `get_blackout_probability_ensemble` en vez de
`get_blackout_probability` en el caso del orchestrator). La hipótesis de la
tormenta de subprocesos queda REFUTADA como causa única: persisten timeouts individuales de 60s con
transporte compartido (1 proceso/server), lo que apunta a una causa distinta
(posible contención SQLite en gridguard.db o latencia real de alguna tool/API
externa bajo el patrón de llamadas concurrentes del orchestrator). Próximo paso:
`superpowers:systematic-debugging` con hipótesis de contención SQLite y cleanup de
event loops de ADK (`Error cleaning up session stdio_session: original event loop
is closed` aparece repetidamente antes de cada timeout).
