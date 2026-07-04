# Orchestrator Crash Resilience Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Que un `ValidationError` de pydantic sin capturar en un sub-agente (ADK 2.3.0, `agent_tool.py::run_async`) ya no aborte todo el ciclo de monitorización sin producir resultado.

**Architecture:** Función `_consume_events_safely(result) -> list` en `agents/orchestrator.py` que consume el generador de eventos del Runner en un bucle con `try/except`, conservando lo ya capturado si algo lanza a mitad de camino. Reemplaza los 2 usos de `events = list(result)` (`run()` y `run_scenario()`).

**Tech Stack:** Python puro, sin dependencias nuevas.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-04-orchestrator-crash-resilience-design.md`.
- Trabajar sobre `main` del proyecto `"C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project"`. Prefijar CADA comando con `cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && `.
- `PYTHONUTF8=1` en todos los comandos python.
- `python -m pytest -q` corre solo la suite offline. NO ejecutar tests `-m integration` salvo en la Task 2.
- Commits en español, conventional, con trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- NO modificar la causa raíz en ADK (código de terceros, fuera del repo). NO reintentar la skill que falló (eso ya lo cubre `escalation_enforcer`, que corre después si esa skill queda ausente en la trayectoria).
- Estado actual EXACTO de `agents/orchestrator.py` (verificado): `run()` línea 135
  `events = list(runner.run(session_id=session.id, user_id="gridguard-system", new_message=message))`
  (llamada inline, sin variable `result` intermedia); `run_scenario()` línea 188
  `events = list(result)` (con `result = runner.run(...)` ya asignado antes, línea 181-185).

---

### Task 1: `_consume_events_safely` + tests unitarios

**Files:**
- Modify: `agents/orchestrator.py`
- Test: `tests/agents/test_consume_events_safely.py`

**Interfaces:**
- Produces: `_consume_events_safely(result) -> list` — reemplaza los 2 usos de `list(result)` en `run()` y `run_scenario()`.

- [ ] **Step 1: Escribir los tests que fallan**

Crear `tests/agents/test_consume_events_safely.py`:

```python
"""_consume_events_safely: no dejar que una excepcion a mitad del generador
de Events del Runner aborte el ciclo completo sin producir resultado.

Spec: docs/superpowers/specs/2026-07-04-orchestrator-crash-resilience-design.md
"""
from __future__ import annotations

from agents.orchestrator import _consume_events_safely


def test_generador_normal_sin_excepcion():
    def _gen():
        yield "a"
        yield "b"
        yield "c"

    assert _consume_events_safely(_gen()) == ["a", "b", "c"]


def test_generador_lanza_a_mitad_conserva_lo_ya_capturado():
    def _gen():
        yield "a"
        yield "b"
        raise ValueError("simulacion de ValidationError de ADK")

    assert _consume_events_safely(_gen()) == ["a", "b"]


def test_generador_lanza_inmediatamente_devuelve_lista_vacia():
    def _gen():
        raise ValueError("simulacion de fallo inmediato")
        yield  # pragma: no cover — nunca se alcanza, hace de esto un generador

    assert _consume_events_safely(_gen()) == []


def test_generador_vacio():
    def _gen():
        return
        yield  # pragma: no cover

    assert _consume_events_safely(_gen()) == []
```

- [ ] **Step 2: Verificar que fallan**

Run: `cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && PYTHONUTF8=1 python -m pytest tests/agents/test_consume_events_safely.py -v`
Expected: todos ERROR/FAILED con `ImportError: cannot import name '_consume_events_safely'`

- [ ] **Step 3: Implementar `_consume_events_safely` y aplicarla en `run()`/`run_scenario()`**

En `agents/orchestrator.py`, añadir la función justo antes de `def run(region...)`:

```python
def _consume_events_safely(result) -> list:
    """Consume el generador de Events del Runner, tolerando que un sub-agente
    (AgentTool con output_schema estricto) lance una excepcion no capturada
    por ADK 2.3.0 (ej. pydantic.ValidationError en agent_tool.py::run_async
    cuando la skill responde texto libre en vez de JSON). Sin esto, UN turno
    mal formateado aborta TODO el ciclo de monitorizacion sin producir ningun
    resultado; con esto, se degrada a "esa skill no produjo output valido" y
    el flujo continua (incluido escalation_enforcer, que puede reparar el hueco).
    """
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

En `run()` (línea 134-139 actual), reemplazar:

```python
    # ADK 2.3.0: runner.run() returns a Generator[Event, None, None] — consume it.
    events = list(runner.run(
        session_id=session.id,
        user_id="gridguard-system",
        new_message=message,
    ))
```

por:

```python
    # ADK 2.3.0: runner.run() returns a Generator[Event, None, None] — consume it.
    events = _consume_events_safely(runner.run(
        session_id=session.id,
        user_id="gridguard-system",
        new_message=message,
    ))
```

En `run_scenario()` (línea 187-188 actual), reemplazar:

```python
    # ADK 2.3.0: runner.run() returns a Generator[Event, None, None] — consume it.
    events = list(result)
```

por:

```python
    # ADK 2.3.0: runner.run() returns a Generator[Event, None, None] — consume it.
    events = _consume_events_safely(result)
```

- [ ] **Step 4: Verificar que pasan**

Run: `cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && PYTHONUTF8=1 python -m pytest tests/agents/test_consume_events_safely.py -v`
Expected: 4 passed

- [ ] **Step 5: Suite offline completa**

Run: `cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && PYTHONUTF8=1 python -m pytest -q`
Expected: 296 passed, 16 deselected (292 previos + 4 nuevos)

- [ ] **Step 6: Commit**

```bash
git add agents/orchestrator.py tests/agents/test_consume_events_safely.py
git commit -m "fix(orchestrator): _consume_events_safely — no abortar el ciclo si un sub-agente lanza ValidationError

ADK 2.3.0 no captura la ValidationError de pydantic cuando una skill
AgentTool con output_schema estricto responde texto libre; sin esto,
un solo turno mal formateado abortaba TODO el ciclo (list(result)
propaga la excepcion sin devolver los eventos ya capturados).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Validación live (repetir la corrida dirigida) + docs

**Files:**
- Modify: `eval/README.md`

**Interfaces:**
- Consumes: `_consume_events_safely` ya wireada (Task 1).

**ATENCIÓN:** consume cuota Gemini real (~1-2 min, la misma corrida dirigida de 2 tests que
ya se ejecutó en la Task 3 del plan `escalation-enforcer`). Ejecutar UNA sola vez.

- [ ] **Step 1: Repetir la corrida dirigida**

Run (timeout 10 min):

```bash
cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && PYTHONUTF8=1 python -m pytest "eval/agents/orchestrator/test_trajectory.py::TestBlackoutTrajectory::test_run_scenario_critical" "eval/agents/orchestrator/test_trigger.py::TestOrchestratorLiveTriggers::test_critical_scenario_triggers_full_escalation" -m integration -v > ".superpowers/sdd/eval_live_resilience_targeted.log" 2>&1
```

- [ ] **Step 2: Analizar el resultado**

```bash
grep -E "PASSED|FAILED" .superpowers/sdd/eval_live_resilience_targeted.log
grep -c "ValidationError" .superpowers/sdd/eval_live_resilience_targeted.log
grep -c "Runner.run() interrumpido" .superpowers/sdd/eval_live_resilience_targeted.log
```

Verificar: si aparece "Runner.run() interrumpido" en el log, confirma que
`_consume_events_safely` está actuando (el crash ocurrió pero ya no propaga). Buscar
también evidencia de que `enforce_escalation_plan` corrió: los tool names con
`"args": {"forced": true}` en la trayectoria del resultado, si el test imprime el dict
de trayectoria en el traceback de un assert fallido.

- [ ] **Step 3: Documentar el resultado medido**

Ampliar la sección "Enforcer determinista de la escalación" en `eval/README.md` con un
subapartado "Resiliencia ante ValidationError (2026-07-04)": el resultado real (PASSED/
FAILED de cada uno), si el enforcer llegó a ejecutarse esta vez, y si no, qué se observó
en el log. Documentar con honestidad — si sigue en rojo por otra causa, describirla tal
cual sin suavizar.

- [ ] **Step 4: Commit y push**

```bash
git add eval/README.md
git commit -m "docs(eval): resultado de la validacion tras la resiliencia ante ValidationError

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push
```
