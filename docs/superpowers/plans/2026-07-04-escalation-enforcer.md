# Escalation Enforcer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Garantizar por código (no por prompt) que en `run_scenario()` toda skill/sub-agente que `escalation_planner` marque como obligatoria se ejecute, aunque el LLM del orquestador se detenga antes.

**Architecture:** Módulo nuevo `agents/escalation_enforcer.py` con `enforce_escalation_plan(result, trajectory, snapshot)`: recalcula el plan determinista con los valores ya presentes en el resultado final, detecta huecos contra la trayectoria real, e invoca por código (runner aislado, sin pasar por el LLM del orquestador) lo que falte. Wireado como último paso de `run_scenario()`.

**Tech Stack:** Python puro para la detección de huecos; `google.adk.runners.Runner` + `InMemorySessionService` (mismo patrón que `eval/agents/_harness.py::run_agent_live`) para invocar skills/sub-agentes de forma aislada cuando faltan.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-04-escalation-enforcer-design.md`.
- Trabajar sobre `main` del proyecto `"C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project"`. Prefijar CADA comando con `cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && `.
- `PYTHONUTF8=1` en todos los comandos python.
- NUNCA leer `.env`, `*.key`, `secrets.*`, `credentials.*`.
- **Alcance decidido con el usuario:** el enforcer SOLO se conecta en `run_scenario()`, NO en `run()`. NO se corrige la sobre-escalación en VERDE (`test_normal_scenario_triggers_nothing` puede seguir en rojo — residual conocido, documentado, NO marcado `flaky`). NO se toca `prompts/orchestrator.md` ni `tools/escalation_planner.py` (ambos ya validados).
- **Bubbling de trayectoria (verificado contra el código fuente de ADK 2.3.0, ver spec):** `AnomalyDetectorSkill`, `RenewablesForecastSkill`, `RegulatoryComplianceSkill` son `AgentTool` — su `run_async` ejecuta un `Runner` anidado y descarta sus eventos internos (`agent_tool.py:276-291`: itera `event` en un bucle async pero solo actualiza `state_delta`/`last_content`, nunca los reemite). Al forzarlas, añadir SOLO una entrada sintética `{"tool": "<nombre_skill>", "args": {"forced": True}}`. `cross_border_agent` y `report_agent` son `sub_agents` (transferencia ADK vía `_find_agent_to_run` en `runners.py`, misma invocación) — sus tool-calls SÍ burbujean en el flujo normal. Al forzarlos, añadir la entrada sintética **Y** todas las entradas de `forced["trajectory"]` (el resultado de `_run_agent_once`).
- `python -m pytest -q` corre solo la suite offline (pytest.ini deselecciona `integration`). NO ejecutar tests `-m integration` salvo que la task lo indique explícitamente.
- Commits en español, conventional, con trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Nombres de campo exactos del resultado del orquestador (`FORMATO DE RESPUESTA FINAL` de `prompts/orchestrator.md`): `timestamp`, `risk_score`, `blackout_probability`, `markov_state` (string `"S0".."S4"`), `severity`, `scope`, `alert`, `anomaly_report`, `renewables_forecast`, `compliance_report`, `cross_border`, `actions_taken`. `escalation_planner` devuelve: `band`, `mandatory_skills` (list[str]), `cross_border_mode` (`"none"|"assessment"|"emergency"`), `invoke_report_agent` (bool), `emit_alert` (`"none"|"warning"|"critical"`), `scope`, `reason`.

---

### Task 1: Módulo `agents/escalation_enforcer.py` con tests unitarios (sin LLM)

**Files:**
- Create: `agents/escalation_enforcer.py`
- Test: `tests/agents/test_escalation_enforcer.py`

**Interfaces:**
- Consumes: `escalation_planner` de `tools/escalation_planner.py` (firma `escalation_planner(blackout_probability: float, markov_state: int) -> dict`); `alert_formatter` de `tools/alert_formatter.py` (firma `alert_formatter(risk: dict, anomalies: list[dict], scope: str = "peninsular") -> dict`); los agentes `AnomalyDetectorSkill.agent`, `RenewablesForecastSkill.agent`, `RegulatoryComplianceSkill.agent` (de `agents/anomaly_detector.py`, `agents/renewables_forecast.py`, `agents/regulatory_compliance.py`), y `cross_border_agent`, `report_agent` (de `agents/cross_border.py`, `agents/report_agent.py`).
- Produces: `enforce_escalation_plan(result: dict, trajectory: list[dict], snapshot: dict) -> dict` — consumido por Task 2 (wiring en `run_scenario()`).

- [ ] **Step 1: Escribir los tests que fallan**

Crear `tests/agents/test_escalation_enforcer.py`:

```python
"""agents/escalation_enforcer.py: repara por codigo la infra-escalacion del
orchestrator cuando el LLM no ejecuta todo lo que escalation_planner manda.

Spec: docs/superpowers/specs/2026-07-04-escalation-enforcer-design.md
"""
from __future__ import annotations

from agents.escalation_enforcer import enforce_escalation_plan


def _base_result(**overrides) -> dict:
    base = {
        "timestamp": "2026-07-04T10:00:00+00:00",
        "risk_score": 85,
        "blackout_probability": 0.75,
        "markov_state": "S3",
        "severity": "CRITICAL",
        "scope": "nacional",
        "alert": None,
        "anomaly_report": None,
        "renewables_forecast": None,
        "compliance_report": None,
        "cross_border": None,
        "actions_taken": ["ree_poller", "risk_scorer"],
    }
    base.update(overrides)
    return base


def test_rojo_fuerza_las_skills_y_subagentes_faltantes(monkeypatch):
    calls = []

    def _fake_run_agent_once(agent, message_text):
        calls.append(getattr(agent, "name", str(agent)))
        name = getattr(agent, "name", "")
        if name == "anomaly_detector_skill":
            output = {"anomalies": [], "severity": "HIGH", "confidence": 0.8,
                       "ml_score": 0.7, "regulatory_context": "x", "reasoning": "x"}
        elif name == "renewables_forecast_skill":
            output = {"risk_level": "HIGH", "expected_generation_pct": 60.0,
                       "critical_window": "14:00-18:00 UTC", "ml_forecast_mw": 30000.0,
                       "reasoning": "x"}
        elif name == "regulatory_compliance_skill":
            output = {"compliance_status": "warning", "reasoning": "x"}
        elif name == "cross_border_agent":
            output = {"can_import": True, "available_capacity_mw": 500.0,
                       "countries": ["FR"], "risk_mitigation": "x"}
        else:  # report_agent
            output = {"raw": "informe redactado"}
        trajectory = [{"tool": f"{name}_internal_tool", "args": {}}] if name in (
            "cross_border_agent", "report_agent") else []
        return {"output": output, "trajectory": trajectory}

    monkeypatch.setattr(
        "agents.escalation_enforcer._run_agent_once", _fake_run_agent_once
    )

    result = _base_result()
    trajectory = [{"tool": "anomaly_detector_skill", "args": {}}]  # el LLM solo llego hasta aqui

    out = enforce_escalation_plan(result, trajectory, snapshot={"demand_mw": 32000})

    tools = {step["tool"] for step in out["trajectory"] if not step["tool"].endswith("_internal_tool")}
    assert "renewables_forecast_skill" in tools
    assert "regulatory_compliance_skill" in tools
    assert "cross_border_agent" in tools
    assert "report_agent" in tools
    # anomaly_detector_skill ya estaba, NO debe forzarse de nuevo
    assert "anomaly_detector_skill" not in calls

    assert out["renewables_forecast"]["risk_level"] == "HIGH"
    assert out["compliance_report"]["compliance_status"] == "warning"
    assert out["cross_border"]["can_import"] is True
    assert any("forzado por enforcer" in a for a in out["actions_taken"])

    # Bubbling: las tool-calls internas de los sub_agents forzados (no de las skills) aparecen
    internal_tools = {step["tool"] for step in out["trajectory"]}
    assert "cross_border_agent_internal_tool" in internal_tools
    assert "report_agent_internal_tool" in internal_tools


def test_verde_no_fuerza_nada():
    result = _base_result(
        risk_score=5, blackout_probability=0.03, markov_state="S0",
        severity="LOW", scope="peninsular",
    )
    trajectory = []

    out = enforce_escalation_plan(result, trajectory, snapshot={})

    assert out["trajectory"] == []
    assert out["anomaly_report"] is None
    assert not any("forzado por enforcer" in a for a in out["actions_taken"])


def test_naranja_con_trayectoria_completa_es_idempotente(monkeypatch):
    def _boom(agent, message_text):
        raise AssertionError("no deberia forzar nada: la trayectoria ya esta completa")

    monkeypatch.setattr("agents.escalation_enforcer._run_agent_once", _boom)

    result = _base_result(
        risk_score=45, blackout_probability=0.40, markov_state="S1",
        severity="MEDIUM", scope="peninsular",
        alert={"severity": "MEDIUM", "scope": "peninsular", "timestamp": "x",
               "summary": "x", "details": "x", "recommended_action": "x"},
    )
    trajectory = [
        {"tool": "anomaly_detector_skill", "args": {}},
        {"tool": "renewables_forecast_skill", "args": {}},
        {"tool": "regulatory_compliance_skill", "args": {}},
        {"tool": "cross_border_agent", "args": {}},
    ]

    out = enforce_escalation_plan(result, trajectory, snapshot={})

    assert out["trajectory"] == trajectory  # sin cambios


def test_alerta_se_rellena_con_alert_formatter_puro_sin_llm():
    result = _base_result(
        risk_score=85, blackout_probability=0.75, markov_state="S3",
        severity="CRITICAL", scope="nacional", alert=None,
        anomaly_report={"anomalies": [{"feature": "demand_mw", "value": 32000.0,
                                        "deviation_sigma": 3.2, "description": "pico"}],
                        "severity": "HIGH", "confidence": 0.8, "ml_score": 0.7,
                        "regulatory_context": "x", "reasoning": "x"},
    )
    trajectory = [
        {"tool": "anomaly_detector_skill", "args": {}},
        {"tool": "renewables_forecast_skill", "args": {}},
        {"tool": "regulatory_compliance_skill", "args": {}},
        {"tool": "cross_border_agent", "args": {}},
        {"tool": "report_agent", "args": {}},
    ]

    out = enforce_escalation_plan(result, trajectory, snapshot={})

    assert out["alert"] is not None
    assert out["alert"]["severity"] == "CRITICAL"
    assert out["alert"]["scope"] == "nacional"


def test_markov_state_mal_formado_no_lanza_excepcion():
    result = _base_result(markov_state="?", blackout_probability=0.5)
    out = enforce_escalation_plan(result, trajectory=[], snapshot={})
    assert isinstance(out, dict)  # no lanza
```

- [ ] **Step 2: Verificar que fallan**

Run: `cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && PYTHONUTF8=1 python -m pytest tests/agents/test_escalation_enforcer.py -v`
Expected: todos ERROR con `ModuleNotFoundError: No module named 'agents.escalation_enforcer'`

- [ ] **Step 3: Implementar el enforcer**

Crear `agents/escalation_enforcer.py`:

```python
"""agents/escalation_enforcer.py — repara por codigo la infra-escalacion.

El LLM del orchestrator no obedece consistentemente el resultado VINCULANTE
de escalation_planner (verificado en la corrida live 2026-07-04): en ROJO se
detiene antes de invocar todo lo obligatorio. Este modulo recalcula el plan
determinista con los valores YA presentes en el resultado final, detecta que
falta contra la trayectoria real, y lo invoca por codigo -- sin pasar por el
LLM del orchestrator, sin tocar el prompt.

Alcance (spec 2026-07-04-escalation-enforcer-design.md): repara SOLO
infra-escalacion (huecos en NARANJA/ROJO). NO deshace sobre-escalacion (VERDE
disparando de mas) -- eso es un residual de no-determinismo LLM aceptado, no
un riesgo de seguridad.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from agents.orchestrator import _extract_result, _extract_trajectory
from tools.alert_formatter import alert_formatter
from tools.escalation_planner import escalation_planner

_SUB_AGENT_NAMES = {"cross_border_agent", "report_agent"}


def _run_agent_once(agent, message_text: str) -> dict[str, Any]:
    """Ejecuta `agent` de forma aislada (runner propio) y devuelve
    {"output": dict, "trajectory": list[{"tool": str, ...}]}.

    Mismo patron que eval/agents/_harness.py::run_agent_live, duplicado
    deliberadamente para no crear una dependencia inversa agents/ -> eval/.
    """
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="gridguard_enforcer", session_service=session_service)
    session = asyncio.run(
        session_service.create_session(app_name="gridguard_enforcer", user_id="enforcer")
    )
    message = genai_types.Content(role="user", parts=[genai_types.Part(text=message_text)])
    events = list(runner.run(session_id=session.id, user_id="enforcer", new_message=message))
    return {"output": _extract_result(events), "trajectory": _extract_trajectory(events)}


def _markov_state_int(label: str) -> int:
    try:
        return int(str(label).lstrip("S"))
    except (ValueError, TypeError):
        return 0


def _force(agent, name: str, message: str, result: dict, out_trajectory: list[dict]) -> dict:
    """Invoca `agent` forzado, fusiona su trayectoria (con bubbling si aplica)
    y devuelve su output. No modifica `result` directamente (lo hace el llamador)."""
    forced = _run_agent_once(agent, message)
    out_trajectory.append({"tool": name, "args": {"forced": True}})
    if name in _SUB_AGENT_NAMES:
        # Bubbling: los sub_agents ya burbujean sus tool-calls internas en el
        # flujo normal (transferencia ADK); replicarlo aqui para que asserts
        # como el de RAG (find_similar_incident dentro de report_agent) pasen.
        out_trajectory.extend(forced["trajectory"])
    result["actions_taken"].append(f"{name} (forzado por enforcer)")
    return forced["output"]


def enforce_escalation_plan(result: dict, trajectory: list[dict], snapshot: dict) -> dict:
    """Garantiza por codigo que todo lo que escalation_planner marca como
    obligatorio se ejecuto, invocando lo que falte de forma aislada.

    Args:
        result: dict de resultado del orchestrator (FORMATO DE RESPUESTA FINAL).
        trajectory: lista de {"tool": str, "args": dict} ya capturada del run del LLM.
        snapshot: GridSnapshot (dict) del ciclo actual, para construir los mensajes
            de las invocaciones forzadas.

    Returns:
        `result` mutado (mismas claves), con `result["trajectory"]` actualizado.
    """
    p = result.get("blackout_probability", 0.0) or 0.0
    state = _markov_state_int(result.get("markov_state", "S0"))
    plan = escalation_planner(p, state)

    trajectory = list(trajectory)  # copia local, no mutar el arg original
    tool_names = {step["tool"] for step in trajectory}

    context_json = json.dumps({
        "snapshot": snapshot,
        "risk_score": result.get("risk_score"),
        "blackout_probability": p,
        "markov_state": result.get("markov_state"),
    })

    skill_agents = {
        "anomaly_detector_skill": ("agents.anomaly_detector", "AnomalyDetectorSkill"),
        "renewables_forecast_skill": ("agents.renewables_forecast", "RenewablesForecastSkill"),
        "regulatory_compliance_skill": ("agents.regulatory_compliance", "RegulatoryComplianceSkill"),
    }
    result_key_by_skill = {
        "anomaly_detector_skill": "anomaly_report",
        "renewables_forecast_skill": "renewables_forecast",
        "regulatory_compliance_skill": "compliance_report",
    }

    for skill_name in plan["mandatory_skills"]:
        if skill_name in tool_names:
            continue
        module_name, attr_name = skill_agents[skill_name]
        import importlib
        skill_tool = getattr(importlib.import_module(module_name), attr_name)
        message = f"[ENFORCER] Analiza este contexto y genera tu output: {context_json}"
        output = _force(skill_tool.agent, skill_name, message, result, trajectory)
        result[result_key_by_skill[skill_name]] = output
        tool_names.add(skill_name)

    if plan["cross_border_mode"] != "none" and "cross_border_agent" not in tool_names:
        from agents.cross_border import cross_border_agent
        message = (
            f"[ENFORCER] mode='{plan['cross_border_mode']}'. "
            f"Evalua la capacidad de importacion. Contexto: {context_json}"
        )
        output = _force(cross_border_agent, "cross_border_agent", message, result, trajectory)
        result["cross_border"] = output
        tool_names.add("cross_border_agent")

    if plan["invoke_report_agent"] and "report_agent" not in tool_names:
        from agents.report_agent import report_agent
        message = (
            f"[ENFORCER] report_type='emergency'. Redacta el informe de emergencia. "
            f"Contexto: {context_json}"
        )
        _force(report_agent, "report_agent", message, result, trajectory)
        tool_names.add("report_agent")

    if plan["emit_alert"] != "none" and result.get("alert") is None:
        anomaly_report = result.get("anomaly_report") or {}
        risk = {
            "level": result.get("severity", "LOW"),
            "value": result.get("risk_score", 0),
            "blackout_probability": p,
        }
        result["alert"] = alert_formatter(
            risk=risk,
            anomalies=anomaly_report.get("anomalies", []),
            scope=plan["scope"],
        )

    result["trajectory"] = trajectory
    return result
```

- [ ] **Step 4: Verificar que pasan**

Run: `cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && PYTHONUTF8=1 python -m pytest tests/agents/test_escalation_enforcer.py -v`
Expected: 5 passed

- [ ] **Step 5: Suite offline completa**

Run: `cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && PYTHONUTF8=1 python -m pytest -q`
Expected: 292 passed, 16 deselected (287 previos + 5 nuevos)

- [ ] **Step 6: Commit**

```bash
git add agents/escalation_enforcer.py tests/agents/test_escalation_enforcer.py
git commit -m "feat(orchestrator): escalation_enforcer — repara por codigo la infra-escalacion del LLM

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Wiring en `run_scenario()` + documentación de la limitación

**Files:**
- Modify: `agents/orchestrator.py:161-190` (función `run_scenario`)
- Modify: `eval/README.md`

**Interfaces:**
- Consumes: `enforce_escalation_plan(result, trajectory, snapshot) -> dict` de `agents/escalation_enforcer.py` (Task 1).
- Produces: `run_scenario()` devuelve el resultado ya reparado; ningún cambio de firma pública.

- [ ] **Step 1: Wiring**

En `agents/orchestrator.py`, añadir el import junto a los demás imports del proyecto:

```python
from agents.escalation_enforcer import enforce_escalation_plan
```

En `run_scenario()`, reemplazar:

```python
    # ADK 2.3.0: runner.run() returns a Generator[Event, None, None] — consume it.
    events = list(result)
    output = _extract_result(events)
    # Agregar trayectoria de tools para los tests de eval
    output["trajectory"] = _extract_trajectory(events)
    return output
```

por:

```python
    # ADK 2.3.0: runner.run() returns a Generator[Event, None, None] — consume it.
    events = list(result)
    output = _extract_result(events)
    # Agregar trayectoria de tools para los tests de eval
    output["trajectory"] = _extract_trajectory(events)
    # Enforcer determinista: repara por codigo lo que el LLM no ejecuto de lo
    # que escalation_planner marco como obligatorio (ver
    # docs/superpowers/specs/2026-07-04-escalation-enforcer-design.md).
    output = enforce_escalation_plan(output, output["trajectory"], snapshot)
    return output
```

(`snapshot` ya está en scope: es la variable definida más arriba en la misma función,
línea `snapshot = scenario.get("snapshots", [{}])[-1]`.)

- [ ] **Step 2: Verificar que el import no crea un ciclo**

`agents/escalation_enforcer.py` importa de `agents.orchestrator` (`_extract_result`,
`_extract_trajectory`), y `agents/orchestrator.py` importará de `agents.escalation_enforcer`
(`enforce_escalation_plan`) — esto ES un ciclo de imports a nivel de módulo. Resolverlo
haciendo el import de `enforce_escalation_plan` DIFERIDO (dentro de la función
`run_scenario()`, no a nivel de módulo):

```python
def run_scenario(scenario: dict) -> dict[str, Any]:
    """..."""
    from agents.escalation_enforcer import enforce_escalation_plan  # import diferido: evita ciclo con escalation_enforcer, que importa de este modulo
    ...
```

NO añadir el import a nivel de módulo (elimínalo de ahí si lo pusiste en el Step 1) —
solo dentro de la función `run_scenario`, justo al principio de su cuerpo.

- [ ] **Step 3: Verificar el ciclo resuelto**

Run: `cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && PYTHONUTF8=1 python -c "from agents.orchestrator import run_scenario; from agents.escalation_enforcer import enforce_escalation_plan; print('OK sin ImportError circular')"`
Expected: `OK sin ImportError circular`

- [ ] **Step 4: Documentar la limitación en `eval/README.md`**

Añadir al final del fichero (tras la última sección existente):

```markdown

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
```

- [ ] **Step 5: Suite offline completa**

Run: `cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && PYTHONUTF8=1 python -m pytest -q`
Expected: 292 passed, 16 deselected

- [ ] **Step 6: Commit**

```bash
git add agents/orchestrator.py eval/README.md
git commit -m "feat(orchestrator): conecta escalation_enforcer en run_scenario (import diferido, evita ciclo)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Validación live dirigida (2 tests, no la corrida completa)

**Files:**
- Modify: `eval/README.md` (resultado medido)

**Interfaces:**
- Consumes: `run_scenario()` ya reparado (Task 2).
- Produces: evidencia de que `test_run_scenario_critical` y
  `test_critical_scenario_triggers_full_escalation` pasan.

**ATENCIÓN:** esta task consume cuota Gemini real (~3-5 min, NO ~10-25 min — es
una corrida dirigida a 2 tests, no la suite completa de 12). NO ejecutar la
corrida completa de `eval/agents -m integration` — sería redundante y cara.

- [ ] **Step 1: Corrida dirigida**

Run (timeout 10 min):

```bash
cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && PYTHONUTF8=1 python -m pytest "eval/agents/orchestrator/test_trajectory.py::TestBlackoutTrajectory::test_run_scenario_critical" "eval/agents/orchestrator/test_trigger.py::TestOrchestratorLiveTriggers::test_critical_scenario_triggers_full_escalation" -m integration -v > ".superpowers/sdd/eval_live_enforcer_targeted.log" 2>&1
```

- [ ] **Step 2: Analizar el resultado**

```bash
grep -E "PASSED|FAILED" .superpowers/sdd/eval_live_enforcer_targeted.log
```

Expected: ambos `PASSED`. Si alguno sigue en rojo, leer el traceback completo del
log (`cat .superpowers/sdd/eval_live_enforcer_targeted.log`) — puede ser un
detalle de la fusión de trayectoria (nombres de agente, bubbling) que necesita
ajuste en `agents/escalation_enforcer.py`. NO repetir la corrida sin antes leer
y entender el fallo concreto (cuota limitada).

- [ ] **Step 3: Documentar el resultado medido en `eval/README.md`**

Reemplazar/ampliar la sección "Enforcer determinista de la escalación (2026-07-04)"
añadida en la Task 2 con el resultado real de este Step 1-2 (PASSED/FAILED
literal, no inventado). Si ambos pasan, indicar explícitamente que
`test_normal_scenario_triggers_nothing` sigue siendo el residual conocido
(sobre-escalación en VERDE) y por qué no se corrigió (ver spec).

- [ ] **Step 4: Commit y push**

```bash
git add eval/README.md
git commit -m "docs(eval): resultado de la validacion dirigida del escalation_enforcer

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push
```
