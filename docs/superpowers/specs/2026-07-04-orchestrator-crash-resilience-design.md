# Spec: resiliencia del orquestador ante ValidationError de sub-agentes (ADK 2.3.0)

**Fecha:** 2026-07-04 · **Estado:** aprobada por el usuario
**Contexto:** la validación live dirigida del `escalation_enforcer` (commit 90f7d1f) reveló
que el enforcer nunca llega a ejecutarse cuando una skill (`AgentTool` con `output_schema`
estricto, ej. `AnomalyDetectorSkill`) responde texto libre en vez de JSON: ADK 2.3.0 no
captura la `pydantic.ValidationError` resultante en `agent_tool.py::run_async` (línea 299,
`validate_schema(output_schema, merged_text)`), la relanza, y sube sin manejar hasta
`list(result)` en `agents/orchestrator.py` (líneas 154 y 188), abortando TODO el ciclo —
`_extract_result`/`_extract_trajectory`/`enforce_escalation_plan` nunca se alcanzan.

Esto es un problema de robustez independiente y más fundamental que la infra-escalación que
`escalation_enforcer` repara: un sistema de monitorización de red eléctrica que se cae por
completo porque UN sub-agente respondió mal formateado es un riesgo real (peor que no escalar
del todo: no produce NINGÚN resultado para ese ciclo).

## Decisión

Envolver el consumo del generador de eventos del `Runner` en `run()` y `run_scenario()` con
manejo de excepción: si `runner.run()` lanza a mitad de iteración, conservar los eventos ya
capturados hasta ese punto y continuar el flujo normal (extracción de resultado + trayectoria
+ enforcer) en vez de abortar la función completa. Degrada el turno fallido a "esa skill no
produjo output válido" en lugar de "todo el ciclo de monitorización falló".

**Fuera de alcance:** no se corrige la causa raíz en ADK (`agent_tool.py`, código de terceros,
no del proyecto); no se reintenta la llamada a la skill que falló (fuera de alcance, ver
`escalation_enforcer` que ya cubre invocarla de nuevo si el plan la marca como obligatoria y
no aparece en la trayectoria).

## Diseño

En `agents/orchestrator.py`, reemplazar el patrón `events = list(result)` (aparece 2 veces:
`run()` línea ~154, `run_scenario()` línea ~188) por una función compartida:

```python
def _consume_events_safely(result) -> list:
    """Consume el generador de Events del Runner, tolerando que un sub-agente
    (AgentTool con output_schema estricto) lance una excepcion no capturada
    por ADK 2.3.0 (ej. pydantic.ValidationError en agent_tool.py::run_async
    cuando la skill responde texto libre en vez de JSON). Sin esto, UN turno
    mal formateado aborta TODO el ciclo de monitorizacion sin producir ningun
    resultado."""
    events: list = []
    try:
        for event in result:
            events.append(event)
    except Exception as exc:
        log.warning(
            "Runner.run() interrumpido por excepcion no capturada (probable "
            "output_schema invalido de una skill): %s. Continuando con %d "
            "eventos ya capturados.", exc, len(events),
        )
    return events
```

Ambas funciones reemplazan `events = list(result)` por `events = _consume_events_safely(result)`.

## Testing

- Unit test: un generador falso que produce N eventos válidos y luego lanza una excepción;
  `_consume_events_safely` debe devolver los N eventos sin propagar la excepción.
- Unit test: un generador que lanza inmediatamente (0 eventos) → devuelve `[]` sin lanzar.
- Unit test: un generador normal sin excepción → equivalente a `list(result)`.
- Suite offline íntegra (292 + nuevos).
- Validación live: repetir la MISMA corrida dirigida de 2 tests de la Task 3 anterior
  (`test_run_scenario_critical`, `test_critical_scenario_triggers_full_escalation`) — una
  sola vez más, para confirmar que ahora sí se alcanza `enforce_escalation_plan`.

## Criterio de éxito

- Los 3 unit tests nuevos verdes, suite offline sin regresiones.
- En la corrida live repetida, al menos evidencia de que `enforce_escalation_plan` se ejecutó
  (aunque el test en sí pueda seguir en rojo por flakiness LLM adicional — lo que importa es
  que el ciclo YA NO aborta con excepción sin producir resultado).
