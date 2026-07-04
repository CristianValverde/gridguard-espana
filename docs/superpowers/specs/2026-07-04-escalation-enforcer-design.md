# Spec: enforcer determinista de la escalación (post-procesado, sin tocar el prompt)

**Fecha:** 2026-07-04 · **Estado:** aprobada por el usuario
**Contexto:** el plan `escalation-planner` (commits 701e422..3f71a60) creó `escalation_planner`
como función pura determinista y reescribió el PASO 2 del prompt para que el LLM ejecute
VINCULANTEMENTE su resultado. La validación live final (commit c53803e) confirmó que el
planner **se invoca correctamente en runtime** (orden y argumentos correctos), pero el LLM
del orquestador **no obedece consistentemente** su output: en escenario ROJO se detiene tras
`anomaly_detector_skill` sin llegar a `cross_border_agent`/`renewables_forecast_skill`/
`regulatory_compliance_skill`/`report_agent`; en VERDE dispara skills que el planner dijo
que no hacían falta.

Dos intentos previos de reforzar el prompt (prosa → "VINCULANTE" explícito) no cambiaron el
comportamiento, lo que descarta framing/redacción como causa y apunta a un límite de la
arquitectura actual (agente único con tool-calling libre, sin garantía de ejecución).

## Decisión

**Post-procesado determinista, sin tocar el prompt.** Tras completar la corrida del LLM en
`run_scenario()`, recalcular el plan definitivo llamando a `escalation_planner()` en Python
puro con los valores YA presentes en el resultado final (`blackout_probability`,
`markov_state`), comparar contra la trayectoria real de tools invocadas, e invocar
POR CÓDIGO (no vía LLM del orquestador) cualquier skill/sub-agente de `mandatory_skills`,
`cross_border_mode` o `invoke_report_agent` que falte. Esto convierte la garantía de
"todo lo obligatorio se ejecuta" de una promesa del prompt (no fiable) a una invariante de
código (determinista, ya no depende de que el LLM obedezca).

**Alcance de esta iteración (decidido con el usuario):**
- Repara **infra-escalación** (el LLM se quedó corto en NARANJA/ROJO) — el caso de seguridad
  real (un sistema que no escala cuando debería es peligroso).
- **NO** repara la **sobre-escalación** (VERDE disparando skills de más, ej.
  `test_normal_scenario_triggers_nothing`): el enforcer no puede deshacer una llamada que el
  LLM ya hizo genuinamente. Se documenta como residual conocido de no-determinismo LLM (sobre-
  cautela, no riesgo de seguridad); el test NO se marca `flaky` (mantiene la honestidad de la
  suite acordada en el plan `escalation-planner`).
- Se implementa **solo en `run_scenario()`** (el path que usan los evals). `run()` (ciclo
  real de producción) queda documentado como limitación conocida: no expone el snapshot de
  red de forma tan directa como `run_scenario()`, y extraerlo desde la trayectoria de
  `function_call` es más frágil — trabajo futuro, no de esta iteración.

## Diseño

### 1. Módulo nuevo `agents/escalation_enforcer.py`

Función principal:

```python
def enforce_escalation_plan(result: dict, trajectory: list[dict], snapshot: dict) -> dict
```

Lógica:

1. Extraer `markov_state_int` de `result["markov_state"]` (formato `"S0".."S4"` → `int` del
   dígito). Si falta o no parsea, usar `0`.
2. `plan = escalation_planner(result.get("blackout_probability", 0.0), markov_state_int)` —
   recomputación determinista, NO se confía en lo que el LLM diga que calculó.
3. `tool_names = {step["tool"] for step in trajectory}`.
4. **Skills faltantes:** para cada `skill` en `plan["mandatory_skills"]` no presente en
   `tool_names`, invocar el agente subyacente vía un runner aislado (`_run_agent_once`, ver
   abajo) con un mensaje que incluya el snapshot; parsear el output (`AnomalyReport`,
   `RenewablesForecast` o `ComplianceReport` según la skill) y fusionarlo en la clave
   correspondiente de `result` (`anomaly_report`, `renewables_forecast`,
   `compliance_report`); añadir a `result["actions_taken"]` una entrada con el sufijo
   `" (forzado por enforcer)"`.

   **Importante — bubbling de trayectoria:** cuando el LLM invoca normalmente
   `anomaly_detector_skill`/`renewables_forecast_skill`/`regulatory_compliance_skill` (son
   `AgentTool`, ver `agents/anomaly_detector.py` etc.), sus tool-calls internas (p.ej.
   `search_regulations`) quedan OCULTAS en la trayectoria del orquestador — solo aparece el
   nombre de la skill como una entrada. Pero `cross_border_agent` y `report_agent` son
   `sub_agents` (transferencia de control ADK, no `AgentTool`): cuando el LLM los invoca
   normalmente, SUS tool-calls internas (`find_similar_incident`, `get_mitigation_procedure`,
   etc.) SÍ aparecen en la trayectoria del orquestador, porque comparten el mismo Runner/
   sesión vía transferencia. El test `test_run_scenario_critical` depende de esto: su assert
   de RAG (`"search_regulations" in tool_names or "find_similar_incident" in tool_names`) solo
   se satisface vía `find_similar_incident`, una tool de `report_agent`.

   Por tanto, al forzar por código: para las 3 Skills (`AgentTool`), añadir a `trajectory`
   ÚNICAMENTE una entrada sintética `{"tool": "<nombre_skill>", "args": {"forced": True}}`
   (igual que aparecería si el LLM la hubiera llamado normalmente — sin bubbling, consistente
   con su naturaleza `AgentTool`). Para `cross_border_agent` y `report_agent` (sub_agents),
   añadir la entrada sintética `{"tool": "<nombre>", "args": {"forced": True}}` **Y ADEMÁS**
   todas las entradas de la trayectoria propia del sub-run forzado (`forced["trajectory"]`,
   el resultado de `_run_agent_once`) — esto reproduce el bubbling que el flujo normal ya
   tiene, y es lo que permite que el assert de RAG pase cuando `report_agent` se fuerza.
5. **cross_border:** si `plan["cross_border_mode"] != "none"` y `"cross_border_agent"` no
   está en `tool_names`, invocar `cross_border_agent` con un mensaje que indique el modo
   (`assessment`/`emergency`) y el snapshot; fusionar en `result["cross_border"]`; aplicar el
   bubbling de trayectoria descrito arriba (es un sub_agent).
6. **report_agent:** si `plan["invoke_report_agent"]` y `"report_agent"` no está en
   `tool_names`, invocar `report_agent` con `report_type="emergency"` y contexto (risk,
   snapshot); no se parsea salida estructurada (report_agent escribe ficheros, no devuelve
   JSON validado) — solo se registra en trayectoria (con bubbling, ver arriba) y
   `actions_taken`.
7. **Alerta:** si `plan["emit_alert"] != "none"` y `result.get("alert")` es `None`, construir
   la alerta con `tools.alert_formatter.alert_formatter` (Python puro, sin LLM) usando
   `risk={"level": result.get("severity", "LOW"), "value": result.get("risk_score", 0),
   "blackout_probability": result.get("blackout_probability", 0.0)}` (mapeo exacto de las
   claves del FORMATO DE RESPUESTA FINAL — `result` NO tiene una clave `"level"` ni `"value"`,
   son `"severity"` y `"risk_score"`), `anomalies=result.get("anomaly_report", {}).get("anomalies", [])`
   (o `[]` si `anomaly_report` es `None`), y `scope=plan["scope"]`.
8. Devuelve `result` mutado (mismas claves que ya definía `FORMATO DE RESPUESTA FINAL`).

Helper privado (duplicado deliberadamente de `eval/agents/_harness.py::run_agent_live` para no
crear una dependencia inversa `agents/` → `eval/`; es ~10 líneas, YAGNI antes que reorganizar
módulos a 2 días del deadline):

```python
def _run_agent_once(agent, message_text: str) -> dict
```

Mismo patrón: `InMemorySessionService` + `Runner` + una sesión + un mensaje +
`_extract_result`/`_extract_trajectory` de `agents.orchestrator`.

### 2. Wiring en `agents/orchestrator.py::run_scenario()`

Tras la línea actual `output["trajectory"] = _extract_trajectory(events)`, añadir:

```python
    output = enforce_escalation_plan(output, output["trajectory"], snapshot)
```

Import nuevo: `from agents.escalation_enforcer import enforce_escalation_plan`.

### 3. Testing

1. **Unit tests deterministas (sin LLM) en `tests/agents/test_escalation_enforcer.py`**:
   mockear `_run_agent_once` (monkeypatch) para verificar la lógica de detección de huecos y
   fusión, SIN invocar Gemini:
   - Plan ROJO con trayectoria que solo tiene `anomaly_detector_skill` → el enforcer invoca
     (mock) `renewables_forecast_skill`, `regulatory_compliance_skill`, `cross_border_agent`
     modo emergency y `report_agent`; el resultado final tiene las 4 claves rellenas y
     `actions_taken` las incluye.
   - Plan VERDE con trayectoria vacía → el enforcer NO invoca nada (mandatory_skills=[]).
   - Plan NARANJA con trayectoria completa (nada falta) → el enforcer no invoca nada
     (idempotencia: no repetir llamadas ya hechas).
   - Alerta: plan con `emit_alert="critical"` y `result["alert"] is None` → se rellena vía
     `alert_formatter` (verificar estructura del Alert, no mockear esta parte: es Python puro).
   - markov_state mal formado (`"?"`, `""`) → no lanza excepción, usa `0`.
2. **Suite offline completa** verde (287 + los nuevos).
3. **Validación live dirigida** (no la corrida completa de 12 tests — sería redundante y
   cara): correr SOLO
   `pytest eval/agents/orchestrator/test_trajectory.py::TestBlackoutTrajectory::test_run_scenario_critical
   eval/agents/orchestrator/test_trigger.py::TestOrchestratorLiveTriggers::test_critical_scenario_triggers_full_escalation
   -m integration -v` (los 2 tests que este fix apunta a resolver; ~3-5 min de cuota en vez de
   ~10). `test_normal_scenario_triggers_nothing` se deja fuera de esta corrida dirigida por
   ser el residual conocido y aceptado.

## Criterio de éxito

- `test_run_scenario_critical` y `test_critical_scenario_triggers_full_escalation` en verde.
- `test_normal_scenario_triggers_nothing` puede seguir en rojo (residual documentado, no
  bloqueante para esta spec).
- Suite offline íntegra, sin regresiones.
- `eval/README.md` actualizado con el resultado medido y la limitación de alcance (solo
  `run_scenario()`, no `run()`).

## Fuera de alcance (documentado como limitación, no implementado)

- Extender el enforcer a `run()` (ciclo real de producción).
- Corregir la sobre-escalación en VERDE.
- Cualquier cambio al prompt (`prompts/orchestrator.md`) o a `escalation_planner` en sí
  (ambos ya validados y correctos).
