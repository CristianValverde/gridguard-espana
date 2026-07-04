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

## Enforcer determinista de la escalación (2026-07-04)

`agents/escalation_enforcer.py::enforce_escalation_plan` repara por código la
infra-escalación cuando el LLM del orquestador no ejecuta todo lo que
`escalation_planner` marca como obligatorio (ver hallazgo en la sección
anterior). Recalcula el plan determinista con los valores ya presentes en el
resultado final y, si algo falta, invoca la skill/sub-agente de forma aislada
(sin pasar por el LLM del orquestador ni tocar el prompt).

**Alcance de esta iteración:**
- Conectado SOLO en `run_scenario()` (el path que usan los evals). `run()`
  (ciclo real de producción) no expone el snapshot de red de forma tan directa
  — extraerlo desde la trayectoria de `function_call` queda como trabajo
  futuro.
- Repara únicamente **infra-escalación** (huecos en NARANJA/ROJO). NO deshace
  **sobre-escalación** (VERDE disparando skills de más): eso requeriría
  ocultar una llamada real que el LLM ya hizo, lo cual sería deshonesto — se
  documenta como residual de no-determinismo LLM aceptado (sobre-cautela, no
  riesgo de seguridad), no como bug.

### Resultado medido de la validación dirigida (2026-07-04, 2 tests, LLM real)

Corrida dirigida a los 2 tests que ejercitan `enforce_escalation_plan` end-to-end
(no la suite completa de 12 — ver `.superpowers/sdd/eval_live_enforcer_targeted.log`,
gitignored):

```
python -m pytest eval/agents/orchestrator/test_trajectory.py::TestBlackoutTrajectory::test_run_scenario_critical \
  eval/agents/orchestrator/test_trigger.py::TestOrchestratorLiveTriggers::test_critical_scenario_triggers_full_escalation \
  -m integration -v
```

**Resultado: 0/2 PASSED (2 failed, 58.86s).** El enforcer en sí no se llegó a
poner a prueba en ninguno de los dos casos — ambos fallos ocurren en la
ejecución LLM normal del orchestrator, **antes** de que
`enforce_escalation_plan` se invoque:

- `test_run_scenario_critical` — **FAILED.** `tool_names` capturado fue
  `['data_normalizer']`: el LLM del orchestrator no llamó a ninguna tool de
  ML (`get_anomaly_score` / `get_blackout_probability[_ensemble]`) en esta
  corrida. Es una falla de trayectoria del LLM orchestrator (no-determinismo
  de turno), no relacionada con el enforcer.
- `test_critical_scenario_triggers_full_escalation` — **FAILED.** Traceback:
  `pydantic_core._pydantic_core.ValidationError: 1 validation error for
  AnomalyReport / Invalid JSON: expected ident ... input_value='Necesito un
  snapshot de ... el snapshot de la red.'`, lanzado desde
  `google.adk.tools.agent_tool.py:299 (validate_schema)`. El orchestrator
  invocó `AnomalyDetectorSkill` (AgentTool con `output_schema=AnomalyReport`,
  ver `agents/anomaly_detector.py:45`) dentro de su árbol normal, y el LLM de
  esa skill respondió con texto conversacional en vez de JSON válido según
  su schema. ADK no captura ese `ValidationError` — lo relanza y aborta
  `runner.run()` completo en `agents/orchestrator.py:188` (`list(result)`),
  **antes** de llegar a la línea 195 donde se invoca
  `enforce_escalation_plan`. Por tanto el enforcer nunca se ejecutó en este
  test; el fallo es anterior en la cadena y no está en el alcance de
  `agents/escalation_enforcer.py`.

**Conclusión:** esta iteración no logró ejercitar el código del enforcer con
LLM real — ambos tests tropezaron con inestabilidad pre-existente del árbol
LLM normal (mismo patrón de no-determinismo documentado en la sección
siguiente: el LLM del orchestrator/sub-agentes no siempre obedece
consistentemente su contrato, sea el plan de escalación o el output_schema).
No se repitió la corrida (cuota limitada) tras leer y entender ambos
tracebacks — ver `.superpowers/sdd/ee-task-3-report.md` para el detalle
completo entregado al controlador.

Como en ninguno de los dos casos se alcanzó a ejercitar `enforce_escalation_plan`,
tampoco hay evidencia nueva sobre `test_normal_scenario_triggers_nothing`
(el residual conocido de sobre-escalación en VERDE); sigue sin corregirse
por la razón ya documentada arriba (deshacer una llamada real del LLM sería
deshonesto).

### Resiliencia ante ValidationError (2026-07-04)

Tras el plan `orchestrator-crash-resilience` (`_consume_events_safely` envolviendo
`runner.run()`/`run_scenario()` en `agents/orchestrator.py`, commit `a9f17ea`), se
repitió la **misma** corrida dirigida a los 2 tests de arriba (una sola vez, ver
`.superpowers/sdd/eval_live_resilience_targeted.log`, gitignored):

```
python -m pytest eval/agents/orchestrator/test_trajectory.py::TestBlackoutTrajectory::test_run_scenario_critical \
  eval/agents/orchestrator/test_trigger.py::TestOrchestratorLiveTriggers::test_critical_scenario_triggers_full_escalation \
  -m integration -v
```

**Resultado: 0/2 PASSED (2 failed, 41.34s).** Esta vez **0 ocurrencias** de
`ValidationError` y **0 ocurrencias** de "Runner.run() interrumpido" en el log
completo — el `ValidationError` de pydantic que abortó `runner.run()` en la
corrida anterior (Task EE-3) **no volvió a aparecer** en esta corrida. No es
posible confirmar con esta única muestra si `_consume_events_safely` llegó a
interceptar un crash real (no se repitió el mismo escenario de fallo), pero
tampoco hay evidencia de que el fix haya introducido ninguna regresión.

Ambos tests volvieron a fallar, pero por una causa **distinta y ya conocida**
— el mismo patrón de no-determinismo LLM documentado arriba, no un crash:

- `test_run_scenario_critical` — **FAILED.** `tool_names` capturado:
  `['data_normalizer', 'stability_proxy_calculator', 'get_serve_features']`.
  El LLM del orchestrator, de nuevo, no llamó a ninguna tool de ML
  (`get_anomaly_score` / `get_blackout_probability[_ensemble]`) en esta corrida.
- `test_critical_scenario_triggers_full_escalation` — **FAILED.** Faltan
  `anomaly_detector_skill`, `cross_border_agent`, `regulatory_compliance_skill`,
  `renewables_forecast_skill`, `report_agent`. Sin trazas de `ValidationError`
  ni de ningún traceback de excepción — el orchestrator simplemente no ejecutó
  las skills obligatorias en este turno.

**El enforcer NO llegó a ejecutarse de forma efectiva en ninguno de los dos
casos**, pero por una razón distinta a la de la corrida anterior:
`enforce_escalation_plan` (`agents/escalation_enforcer.py:71`) recalcula
`escalation_planner(p, state)` a partir de `result["blackout_probability"]` y
`result["markov_state"]` — campos que provienen del *output final* del LLM
orchestrator, no de la trayectoria de tools. Como el LLM no invocó ninguna
tool de ML en ninguno de los dos tests, esos campos quedan en sus valores por
defecto (riesgo bajo), `escalation_planner` no marca ninguna skill como
`mandatory_skills`, y el enforcer no tiene nada que forzar. En otras palabras:
el enforcer depende de que el orchestrator haya escrito al menos una
probabilidad/estado coherente en su output final; si el LLM no consulta ML en
absoluto, el enforcer queda ciego por diseño (no es un bug de
`escalation_enforcer.py` — es una limitación conocida de su alcance actual,
ya anticipada en el diseño: "repara infra-escalación", no "sustituye la
falta total de señal de ML").

**Conclusión honesta:** el bug de crash (`ValidationError` sin capturar) que
bloqueaba las corridas anteriores no se reprodujo esta vez — 0 ocurrencias en
el log — y `_consume_events_safely` no muestra ninguna regresión. Pero la
validación end-to-end del enforcer sigue sin completarse: en las 3 corridas
dirigidas hechas hasta ahora (EE-Task3, esta) el camino de ejecución nunca
llegó a un escenario donde el LLM llame a ML pero omita el resto de la cadena
de escalación — el requisito exacto para poner a prueba
`enforce_escalation_plan` en producción. Se necesitaría o bien más corridas
(para capturar por azar un turno donde el LLM sí consulte ML mínimamente) o
bien un test dirigido que fuerce ese trayecto (p. ej. inyectando un resultado
con `blackout_probability` ya alto sin pasar por el LLM), decisión que queda
para el controlador.

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

**Mitigación de flakiness LLM residual (2026-07-04):** `test_low_margin_scenario_triggers_mitigation`,
`test_ample_margin_scenario_skips_mitigation` (cross_border) y `test_consults_ml_and_baseline`
(anomaly_detector) llevan `@pytest.mark.flaky(reruns=1, reruns_delay=5)` — no-determinismo
LLM de subagentes individuales, NO infraestructura (ya eliminada, ver arriba). Los tests
del orchestrator NO llevan este marcador: tras el `escalation_planner` determinista
(spec `2026-07-04-escalation-planner-design.md`) su resultado debe ser estable.

## Validación live compartida: escalation_planner + lessons-memory (2026-07-04)

Corrida única post-implementación completa de ambos planes (`escalation_planner`
Tasks 1-4, commits 701e422/5f617f6/085f8ff/3f71a60; `lessons-memory` Tasks 1-4,
commits 4fba029/c0afe22/65e5091/627cd2a). Baseline offline confirmado antes de la
corrida: **286 passed, 16 deselected** (`python -m pytest -q`).

Comando: `python -m pytest eval/agents -m integration -v`. **Resultado: 7/12
passed, 5 failed, 1 rerun, 555.70s (9m15s). 0 timeouts MCP, 0 cotas disparadas**
— la causa raíz de infraestructura (import lazy de chromadb, IO sin acotar)
sigue eliminada; todos los fallos de esta corrida son de comportamiento del
agente, no de infraestructura.

| Test | Resultado |
|---|---|
| anomaly_detector / test_consults_ml_and_baseline | PASSED |
| anomaly_detector / test_no_regulations_search_on_normal_scenario | PASSED |
| cross_border / test_low_margin_scenario_triggers_mitigation | PASSED |
| cross_border / test_ample_margin_scenario_skips_mitigation | RERUN → FAILED |
| orchestrator / test_run_scenario_critical (trajectory) | FAILED |
| orchestrator / test_markov_trajectory_escalates_on_blackout_scenario | PASSED |
| orchestrator / test_run_scenario_normal_no_escalation (trajectory) | PASSED |
| orchestrator / test_critical_scenario_triggers_full_escalation | FAILED |
| orchestrator / test_normal_scenario_triggers_nothing | FAILED |
| regulatory_compliance / test_consults_regulations | PASSED |
| renewables_forecast / test_consults_weather_and_load_forecast | FAILED |
| report_agent / test_consults_history_before_writing | PASSED |

**Hallazgo principal — el `escalation_planner` SÍ se invoca pero el LLM del
orchestrator no obedece su output vinculante:** en `test_run_scenario_critical`
(escenario ROJO/blackout), la trayectoria capturada fue:

```
['data_normalizer', 'stability_proxy_calculator', 'get_serve_features',
 'get_grid_state', 'get_blackout_probability_ensemble', 'risk_scorer',
 'store_risk_score', 'escalation_planner', 'get_lessons',
 'anomaly_detector_skill', 'store_lesson']
```

El planner determinista (Python puro, 17 tests unitarios offline, sin cambios
necesarios) se ejecuta correctamente y en el orden esperado tras `risk_scorer`.
Pero el orchestrator invoca solo `anomaly_detector_skill` de las skills que el
planner debería mandar (`mandatory_skills`) y se detiene — no llega a
`cross_border_agent`, `renewables_forecast_skill`, `regulatory_compliance_skill`,
`report_agent` ni a RAG (`search_regulations`/`find_similar_incident`). Esto
produce en cascada los 3 fallos del orchestrator:
- `test_run_scenario_critical`: falla el assert de RAG (RAG solo se satisface
  si la escalación llega a regulatory_compliance).
- `test_critical_scenario_triggers_full_escalation`: `missing =
  {'cross_border_agent', 'regulatory_compliance_skill', 'anomaly_detector_skill',
  'report_agent', 'renewables_forecast_skill'}` — el set completo de
  `_SKILL_TOOLS` menos lo disparado.
- `test_normal_scenario_triggers_nothing`: en el escenario VERDE dispara
  `anomaly_detector_skill` cuando el planner debería devolver
  `mandatory_skills=[]` — incoherente con el hallazgo anterior (el LLM parece
  invocar `anomaly_detector_skill` con cierta independencia del veredicto del
  planner en ambos sentidos: de menos en ROJO, de más en VERDE).

**Evidencia de wiring de lecciones en runtime:** en la misma trayectoria de
`test_run_scenario_critical` aparecen `get_lessons` (PASO 1, tras
`escalation_planner`) y `store_lesson` (PASO 3, cierre de ciclo) — el wiring de
memoria de lecciones (`lessons-memory` Tasks 1-4) SÍ se ejecuta end-to-end
contra el LLM real en al menos un ciclo completo. Esto es evidencia positiva
independiente del resultado del planner: no se observaron errores de
`store_lesson`/`get_lessons` en el log (sin trazas de `TypeError` de
serialización de `timestamp`, confirmando el fix de `_utcnow_iso` en vivo).

**Otros 2 fallos, sin relación con el planner (ya caracterizados en corridas
previas, ver secciones anteriores):**
- `cross_border/test_ample_margin_scenario_skips_mitigation`: flakiness LLM ya
  conocida, con `@pytest.mark.flaky(reruns=1)` — RERUN y volvió a fallar
  (`get_mitigation_procedure` llamado con margen holgado). El retry único no
  cubre la tasa de flip real.
- `renewables_forecast/test_consults_weather_and_load_forecast`: trayectoria
  vacía (`[]`) — timeout/flakiness LLM aislado de este agente, sin marcador
  `flaky`; había pasado 3 corridas seguidas antes de esta.

**STATUS: no se parcheó el prompt ni el planner a ciegas.** El planner
determinista queda validado en su propio alcance (unit tests + se ejecuta y
produce `mandatory_skills` en runtime), pero el criterio de aceptación de la
spec (orchestrator seguía el veredicto de punta a punta) **no se cumple
todavía**: el LLM del orchestrator no está honrando de forma consistente la
instrucción "VINCULANTE" del PASO 2 del prompt para encadenar todas las
`mandatory_skills`. Se requiere `superpowers:systematic-debugging` dedicado
sobre el comportamiento del LLM (candidatos a investigar: framing del PASO 2,
límite de turnos/tokens del orchestrator cortando la cadena tras la primera
skill, o necesidad de function-calling forzado en vez de instrucción en
prompt) antes de reintentar una nueva corrida live — cada iteración cuesta
~10 min de cuota Gemini real.
