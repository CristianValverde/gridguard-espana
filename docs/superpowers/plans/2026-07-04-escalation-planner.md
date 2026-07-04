# Escalation Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reemplazar la política de escalación en prosa del orchestrator por una función Python pura y determinista (`escalation_planner`), y absorber el no-determinismo LLM residual de los subagentes con `pytest-rerunfailures`.

**Architecture:** Un Function Tool nuevo (`tools/escalation_planner.py`) calcula, a partir de `(blackout_probability, markov_state)`, el plan exacto de skills/alertas a ejecutar. El prompt del orchestrator se reduce a "ejecuta TODO lo que devuelva el planner"; la fuente de verdad de la política pasa del prompt al código, testeable offline.

**Tech Stack:** Python puro (sin ADK, sin LLM) para el planner; pydantic no se usa aquí (dict simple, sin I/O); pytest-rerunfailures para mitigar flakiness LLM en 3 tests live concretos.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-04-escalation-planner-design.md`.
- Trabajar sobre `main` del proyecto `"C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project"`. Prefijar CADA comando con `cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && `.
- `PYTHONUTF8=1` en todos los comandos python.
- NUNCA leer `.env`, `*.key`, `secrets.*`, `credentials.*`.
- NUNCA hardcodear modelos Gemini — usar `config/model_loader.py::get_model()` (no aplica a este plan: no se tocan modelos).
- Bandas de escalación exactas (verbatim de la spec): VERDE `P < 0.10`, AMARILLO `[0.10, 0.30)`, NARANJA `[0.30, 0.60)`, ROJO `P >= 0.60`.
- Override 1: `markov_state >= 2` con `P < 0.10` → eleva VERDE a AMARILLO.
- Override 2: `markov_state == 1` con `P` en `[0.10, 0.30)` → añade cross_border `assessment` al flujo AMARILLO.
- `markov_state` se recorta a `[0, 4]`; `blackout_probability` se recorta a `[0.0, 1.0]` — nunca lanzar excepción por entrada fuera de rango.
- `python -m pytest -q` corre solo la suite offline (pytest.ini deselecciona `integration`). NO ejecutar tests `-m integration` salvo que la task lo indique explícitamente (consumen cuota Gemini real, ~10 min).
- Commits en español, conventional, con trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- No tocar `agents/instruction_utils.py` (`as_static_instruction`/`load_prompt`), `guardrails/hooks.py` más allá de añadir el nombre del tool a la whitelist, ni las REGLAS NO NEGOCIABLES / FORMATO DE RESPUESTA del prompt del orchestrator.

---

### Task 1: Function Tool `escalation_planner`

**Files:**
- Create: `tools/escalation_planner.py`
- Test: `tests/tools/test_escalation_planner.py`

**Interfaces:**
- Produces: `escalation_planner(blackout_probability: float, markov_state: int) -> dict` con las claves `band`, `mandatory_skills` (list[str]), `cross_border_mode` (`"none"|"assessment"|"emergency"`), `invoke_report_agent` (bool), `emit_alert` (`"none"|"warning"|"critical"`), `scope` (`"peninsular"|"nacional"`), `reason` (str). Task 2 (wiring del orchestrator) y Task 3 (prompt) consumen esta firma y estas claves exactas.

- [ ] **Step 1: Escribir los tests que fallan**

Crear `tests/tools/test_escalation_planner.py`:

```python
"""Tabla de verdad del escalation_planner: shift-left, sin LLM ni ADK.

Spec: docs/superpowers/specs/2026-07-04-escalation-planner-design.md
"""
from __future__ import annotations

import pytest

from tools.escalation_planner import escalation_planner


def test_verde_sin_markov_no_escala():
    plan = escalation_planner(blackout_probability=0.05, markov_state=0)
    assert plan["band"] == "VERDE"
    assert plan["mandatory_skills"] == []
    assert plan["cross_border_mode"] == "none"
    assert plan["invoke_report_agent"] is False
    assert plan["emit_alert"] == "none"


def test_verde_con_markov_s2_eleva_a_amarillo():
    plan = escalation_planner(blackout_probability=0.05, markov_state=2)
    assert plan["band"] == "AMARILLO"
    assert plan["mandatory_skills"] == ["anomaly_detector_skill"]
    assert "Markov" in plan["reason"]


def test_verde_con_markov_s4_tambien_eleva():
    plan = escalation_planner(blackout_probability=0.02, markov_state=4)
    assert plan["band"] == "AMARILLO"


def test_amarillo_sin_override_no_activa_cross_border():
    plan = escalation_planner(blackout_probability=0.20, markov_state=0)
    assert plan["band"] == "AMARILLO"
    assert plan["mandatory_skills"] == ["anomaly_detector_skill"]
    assert plan["cross_border_mode"] == "none"


def test_amarillo_con_markov_s1_activa_cross_border_assessment():
    plan = escalation_planner(blackout_probability=0.20, markov_state=1)
    assert plan["band"] == "AMARILLO"
    assert plan["cross_border_mode"] == "assessment"


def test_naranja_incluye_las_tres_skills_y_cross_border_assessment():
    plan = escalation_planner(blackout_probability=0.45, markov_state=0)
    assert plan["band"] == "NARANJA"
    assert plan["mandatory_skills"] == [
        "anomaly_detector_skill",
        "renewables_forecast_skill",
        "regulatory_compliance_skill",
    ]
    assert plan["cross_border_mode"] == "assessment"
    assert plan["invoke_report_agent"] is False
    assert plan["emit_alert"] == "warning"
    assert plan["scope"] == "peninsular"


def test_rojo_incluye_todo_y_report_agent():
    plan = escalation_planner(blackout_probability=0.75, markov_state=3)
    assert plan["band"] == "ROJO"
    assert plan["mandatory_skills"] == [
        "anomaly_detector_skill",
        "renewables_forecast_skill",
        "regulatory_compliance_skill",
    ]
    assert plan["cross_border_mode"] == "emergency"
    assert plan["invoke_report_agent"] is True
    assert plan["emit_alert"] == "critical"
    assert plan["scope"] == "nacional"


@pytest.mark.parametrize("p,expected_band", [
    (0.0, "VERDE"),
    (0.099999, "VERDE"),
    (0.10, "AMARILLO"),
    (0.299999, "AMARILLO"),
    (0.30, "NARANJA"),
    (0.599999, "NARANJA"),
    (0.60, "ROJO"),
    (1.0, "ROJO"),
])
def test_bordes_exactos_de_banda(p, expected_band):
    plan = escalation_planner(blackout_probability=p, markov_state=0)
    assert plan["band"] == expected_band


def test_probability_fuera_de_rango_se_recorta():
    plan = escalation_planner(blackout_probability=1.5, markov_state=0)
    assert plan["band"] == "ROJO"
    plan_neg = escalation_planner(blackout_probability=-0.5, markov_state=0)
    assert plan_neg["band"] == "VERDE"


def test_markov_state_fuera_de_rango_se_recorta():
    plan = escalation_planner(blackout_probability=0.05, markov_state=99)
    assert plan["band"] == "AMARILLO"  # recortado a 4 (>=2) con P<0.10
    plan_neg = escalation_planner(blackout_probability=0.05, markov_state=-3)
    assert plan_neg["band"] == "VERDE"  # recortado a 0
```

- [ ] **Step 2: Verificar que fallan**

Run: `cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && PYTHONUTF8=1 python -m pytest tests/tools/test_escalation_planner.py -v`
Expected: todos ERROR con `ModuleNotFoundError: No module named 'tools.escalation_planner'`

- [ ] **Step 3: Implementar el planner**

Crear `tools/escalation_planner.py`:

```python
"""tools/escalation_planner.py — Function Tool: política de escalación
determinista (shift-left: Python puro, sin LLM).

Fuente de verdad de PASO 2 del prompt del orchestrator (prompts/orchestrator.md).
Reemplaza la prosa: la política vive aquí, testeable offline.

Spec: docs/superpowers/specs/2026-07-04-escalation-planner-design.md
"""

from __future__ import annotations

_ANOMALY = "anomaly_detector_skill"
_RENEWABLES = "renewables_forecast_skill"
_COMPLIANCE = "regulatory_compliance_skill"


def escalation_planner(blackout_probability: float, markov_state: int) -> dict:
    """Calcula el plan de escalación para un ciclo de monitorización.

    Args:
        blackout_probability: P(blackout) del ensemble ML, se recorta a [0.0, 1.0].
        markov_state: Estado Markov actual S0-S4 (0-4), se recorta a ese rango.

    Returns:
        dict con band, mandatory_skills, cross_border_mode, invoke_report_agent,
        emit_alert, scope, reason.
    """
    p = min(max(float(blackout_probability), 0.0), 1.0)
    state = min(max(int(markov_state), 0), 4)
    reason_parts: list[str] = []

    if p < 0.10:
        band = "VERDE"
    elif p < 0.30:
        band = "AMARILLO"
    elif p < 0.60:
        band = "NARANJA"
    else:
        band = "ROJO"

    # Override 1: Markov S2+ eleva VERDE a AMARILLO aunque P sea baja.
    if band == "VERDE" and state >= 2:
        band = "AMARILLO"
        reason_parts.append(f"Markov S{state} eleva VERDE→AMARILLO pese a P={p:.2f}")

    cross_border_mode = "none"
    if band == "AMARILLO":
        mandatory_skills = [_ANOMALY]
        # Override 2: Markov S1 en banda AMARILLO activa assessment proactivo.
        if state == 1 and 0.10 <= p < 0.30:
            cross_border_mode = "assessment"
            reason_parts.append("Markov S1 en AMARILLO activa cross_border assessment")
    elif band == "NARANJA":
        mandatory_skills = [_ANOMALY, _RENEWABLES, _COMPLIANCE]
        cross_border_mode = "assessment"
    elif band == "ROJO":
        mandatory_skills = [_ANOMALY, _RENEWABLES, _COMPLIANCE]
        cross_border_mode = "emergency"
    else:  # VERDE sin override
        mandatory_skills = []

    invoke_report_agent = band == "ROJO"

    if band == "ROJO":
        emit_alert = "critical"
        scope = "nacional"
    elif band == "NARANJA":
        emit_alert = "warning"
        scope = "peninsular"
    else:
        emit_alert = "none"
        scope = "peninsular"

    if not reason_parts:
        reason_parts.append(f"Banda {band} por P={p:.2f}, Markov S{state}")

    return {
        "band": band,
        "mandatory_skills": mandatory_skills,
        "cross_border_mode": cross_border_mode,
        "invoke_report_agent": invoke_report_agent,
        "emit_alert": emit_alert,
        "scope": scope,
        "reason": "; ".join(reason_parts),
    }
```

- [ ] **Step 4: Verificar que pasan**

Run: `cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && PYTHONUTF8=1 python -m pytest tests/tools/test_escalation_planner.py -v`
Expected: todos passed (17 tests incluyendo los parametrizados)

- [ ] **Step 5: Suite offline completa**

Run: `cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && PYTHONUTF8=1 python -m pytest -q`
Expected: 278 passed, 16 deselected (261 previos + 17 nuevos)

- [ ] **Step 6: Commit**

```bash
git add tools/escalation_planner.py tests/tools/test_escalation_planner.py
git commit -m "feat(orchestrator): escalation_planner determinista (shift-left, reemplaza prosa del PASO 2)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Wiring del planner en el orchestrator y la whitelist

**Files:**
- Modify: `agents/orchestrator.py`
- Modify: `guardrails/hooks.py:108-116` (bloque `_ALLOWED_TOOLS["gridguard_orchestrator"]`)
- Modify: `eval/agents/orchestrator/test_trigger.py:13-21` (`_EXPECTED_TOOLS`)
- Modify: `tests/guardrails/test_allowed_tools.py` (si existe una lista espejo — verificar antes de tocar)

**Interfaces:**
- Consumes: `escalation_planner` de `tools/escalation_planner.py` (Task 1), firma `escalation_planner(blackout_probability: float, markov_state: int) -> dict`.
- Produces: `escalation_planner` disponible como Function Tool en `orchestrator.tools`; whitelist actualizada.

- [ ] **Step 1: Localizar la lista espejo de tools (si existe)**

Run: `cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && grep -rn "gridguard_orchestrator" tests/guardrails/`

Si `tests/guardrails/test_allowed_tools.py` contiene una copia literal del set de tools del orchestrator (además de la que ya se actualiza en el Step 3), anota su ruta y línea — se actualiza en el Step 3 junto con `eval/agents/orchestrator/test_trigger.py`. Si no existe tal copia, continuar sin cambios ahí.

- [ ] **Step 2: Añadir el tool a `agents/orchestrator.py`**

En el bloque de imports del proyecto (junto a `from tools.risk_scorer import risk_scorer`), añadir:

```python
from tools.escalation_planner import escalation_planner
```

En la lista `tools=[...]` del `Agent`, en la Capa 1 (Function Tools), justo después de `risk_scorer`:

```python
        # Capa 1: Function Tools (Python puro, sin LLM)
        ree_poller,
        data_normalizer,
        stability_proxy_calculator,
        risk_scorer,
        escalation_planner,
        alert_formatter,
```

- [ ] **Step 3: Actualizar la whitelist y el test de wiring del orchestrator**

En `guardrails/hooks.py`, dentro de `_ALLOWED_TOOLS["gridguard_orchestrator"]` (línea ~108-116), añadir `"escalation_planner"` al set (cualquier posición, es un `set`):

```python
    "gridguard_orchestrator": {
        "ree_poller", "data_normalizer", "stability_proxy_calculator",
        "risk_scorer", "escalation_planner", "alert_formatter",
        "get_blackout_probability", "store_risk_score",
        "get_serve_features", "get_ensemble_score", "get_blackout_probability_ensemble",
        "anomaly_detector_skill", "renewables_forecast_skill",
        "cross_border_agent", "report_agent",
        "get_grid_state", "store_grid_state", "regulatory_compliance_skill",
    },
```

En `eval/agents/orchestrator/test_trigger.py`, actualizar `_EXPECTED_TOOLS` (línea 13-21) para incluir `"escalation_planner"` (mismo set que arriba, sin `regulatory_compliance_skill` reordenado — solo añadir la entrada nueva, mismo bloque literal):

```python
_EXPECTED_TOOLS = {
    "ree_poller", "data_normalizer", "stability_proxy_calculator",
    "risk_scorer", "escalation_planner", "alert_formatter",
    "get_blackout_probability", "store_risk_score",
    "get_serve_features", "get_ensemble_score", "get_blackout_probability_ensemble",
    "anomaly_detector_skill", "renewables_forecast_skill",
    "cross_border_agent", "report_agent",
    "get_grid_state", "store_grid_state", "regulatory_compliance_skill",
}
```

- [ ] **Step 4: Suite offline completa**

Run: `cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && PYTHONUTF8=1 python -m pytest -q`
Expected: 278 passed, 16 deselected (mismo número que Task 1 — este task no añade tests, solo wiring)

- [ ] **Step 5: Commit**

```bash
git add agents/orchestrator.py guardrails/hooks.py eval/agents/orchestrator/test_trigger.py
git commit -m "feat(orchestrator): conecta escalation_planner (Capa 1) y actualiza whitelist

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Reescribir el PASO 1/PASO 2 del prompt del orchestrator

**Files:**
- Modify: `prompts/orchestrator.md`
- Test: `tests/agents/test_prompt_files.py` (verificar que ya cubre este prompt; si no, no crear tests nuevos — el contrato lo cubren Task 1 offline y Task 4 live)

**Interfaces:**
- Consumes: la política del planner definida en Task 1 (los mismos nombres de campo: `band`, `mandatory_skills`, `cross_border_mode`, `invoke_report_agent`, `emit_alert`, `scope`).
- Produces: prompt actualizado que el LLM sigue en runtime (no verificable por unit test — se valida en la corrida live de Task 5).

- [ ] **Step 1: Reemplazar el PASO 2 completo del prompt**

Editar `prompts/orchestrator.md`. El PASO 1 (líneas 11-21 del fichero actual) gana un sub-paso `1h` justo después de `1g` (`store_risk_score`):

```
  1h. Llama a `escalation_planner` con blackout_probability=<P del paso 1e> y
      markov_state=<state del get_grid_state del paso 1d>. El resultado de este
      tool es VINCULANTE: ejecuta TODO lo que indique, en el PASO 2 de abajo.
```

Reemplazar íntegramente el bloque "PASO 2 -- Escalacion segun P(blackout_1h) Y estado de Markov:" (desde la línea `PASO 2 -- Escalacion...` hasta el final de la sección `[ROJO -- EMERGENCIA]`, es decir todo el contenido entre las líneas 23 y 62 del fichero actual) por:

```
PASO 2 -- Escalacion (VINCULANTE, segun el resultado de escalation_planner):

El escalation_planner del PASO 1h ya calculo la politica exacta. NO reinterpretes
ni recalcules las bandas tu mismo -- ejecuta EXACTAMENTE lo que indique su
resultado, sin omitir ninguna entrada aunque creas que no aporta:

  - Invoca, EN ORDEN, cada tool listado en `mandatory_skills` (puede ser
    `anomaly_detector_skill`, `renewables_forecast_skill`,
    `regulatory_compliance_skill`, en ese orden si estan presentes). Si la
    lista esta vacia, no invoques ninguna Skill.
  - Si `cross_border_mode` es "assessment", invoca `cross_border_agent` en modo
    assessment. Si es "emergency", invocalo en modo emergencia para evaluar
    importacion de urgencia. Si es "none", no lo invoques.
  - Si `invoke_report_agent` es true, invoca `report_agent` con
    report_type="emergency" DESPUES de las skills y del cross_border_agent.
  - Si `emit_alert` es "warning" o "critical", llama a `alert_formatter` con
    severity segun el RiskScore.level, scope=<scope del planner>, las
    anomalias del AnomalyReport (si se invoco anomaly_detector_skill) y la
    capacidad de importacion disponible (si se invoco cross_border_agent).
    Emite la alerta con esa severity y scope.
  - Si `emit_alert` es "none" y `mandatory_skills` esta vacio: logging pasivo,
    sin invocar Skills/Sub-agents. Mensaje: "Sistema en operacion normal.
    P={P:.1%}, Risk={risk}/100."
  - En `actions_taken` del JSON final, lista TODAS las tools/skills/sub-agentes
    que ejecutaste en este ciclo, incluyendo `escalation_planner`.
```

- [ ] **Step 2: Verificar que el fichero sigue siendo consumido correctamente**

Run: `cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && PYTHONUTF8=1 python -m pytest tests/agents/test_prompt_files.py tests/agents/test_instruction_templating.py -v`
Expected: todos passed (el loader `load_prompt`/`as_static_instruction` no valida contenido semántico, solo que el fichero existe y se carga sin KeyError de templating — confirmar que sigue siendo así)

- [ ] **Step 3: Suite offline completa**

Run: `cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && PYTHONUTF8=1 python -m pytest -q`
Expected: 278 passed, 16 deselected

- [ ] **Step 4: Commit**

```bash
git add prompts/orchestrator.md
git commit -m "feat(orchestrator): PASO 2 del prompt delega la escalacion en escalation_planner

El prompt ya no contiene la politica en prosa (bandas, overrides, listas
por tramo); el LLM ejecuta VINCULANTEMENTE el resultado del planner.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: pytest-rerunfailures para el no-determinismo residual de subagentes

**Files:**
- Modify: `requirements.txt`
- Modify: `eval/agents/cross_border/test_trigger.py:44-77` (clase `TestCrossBorderLiveTriggers`, ambos métodos)
- Modify: `eval/agents/anomaly_detector/test_trigger.py` (método `test_consults_ml_and_baseline`)
- Modify: `eval/README.md` (nota sobre reruns)

**Interfaces:**
- Consumes: nada de tasks anteriores (independiente).
- Produces: marcador `@pytest.mark.flaky(reruns=1, reruns_delay=5)` en 3 tests concretos; no afecta a ningún test del orchestrator.

- [ ] **Step 1: Instalar la dependencia**

Añadir a `requirements.txt`, junto a la línea `pytest>=8.0` (buscar con `grep -n "pytest>=8.0" requirements.txt`):

```
pytest-rerunfailures>=14.0
```

Run: `cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && pip install "pytest-rerunfailures>=14.0"`
Expected: `Successfully installed pytest-rerunfailures-...`

- [ ] **Step 2: Verificar que el plugin no rompe la suite offline**

Run: `cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && PYTHONUTF8=1 python -m pytest -q`
Expected: 278 passed, 16 deselected (el plugin instalado sin marcadores `flaky` no cambia el comportamiento por defecto)

- [ ] **Step 3: Marcar los 3 tests con flakiness LLM conocida**

En `eval/agents/cross_border/test_trigger.py`, en la clase `TestCrossBorderLiveTriggers` (línea 44), añadir el decorador a AMBOS métodos de test (`test_low_margin_scenario_triggers_mitigation` y `test_ample_margin_scenario_skips_mitigation`):

```python
@pytest.mark.integration
class TestCrossBorderLiveTriggers:
    _PROMPT_EMERGENCY = (
        "[EVAL] mode='emergency'. Evalua si Espana puede importar potencia de "
        "emergencia ahora mismo. Snapshot: "
        + json.dumps({"demand_mw": 30200, "blackout_probability": 0.72,
                      "capacity_margin_pct": 3})
    )

    _PROMPT_ASSESSMENT = (
        "[EVAL] mode='assessment'. Evalua proactivamente la capacidad de "
        "importacion disponible sin urgencia. Snapshot: "
        + json.dumps({"demand_mw": 30200, "blackout_probability": 0.15,
                      "capacity_margin_pct": 35})
    )

    @pytest.mark.flaky(reruns=1, reruns_delay=5)
    def test_low_margin_scenario_triggers_mitigation(self):
        from agents.cross_border import cross_border_agent
        from eval.agents._harness import run_agent_live
        result = run_agent_live(cross_border_agent, self._PROMPT_EMERGENCY)
        tools = [step["tool"] for step in result["trajectory"]]
        assert "get_cross_border_flows" in tools, "trigger de flujos ausente"
        assert "get_capacity_margin" in tools, "trigger de margen de capacidad ausente"
        assert "get_mitigation_procedure" in tools, "trigger de procedimiento de mitigacion ausente"

    @pytest.mark.flaky(reruns=1, reruns_delay=5)
    def test_ample_margin_scenario_skips_mitigation(self):
        from agents.cross_border import cross_border_agent
        from eval.agents._harness import run_agent_live
        result = run_agent_live(cross_border_agent, self._PROMPT_ASSESSMENT)
        tools = [step["tool"] for step in result["trajectory"]]
        assert "get_cross_border_flows" in tools, "trigger de flujos ausente"
        assert "get_mitigation_procedure" not in tools, (
            "no deberia buscar procedimiento de mitigacion con margen holgado"
        )
```

En `eval/agents/anomaly_detector/test_trigger.py`, localizar `test_consults_ml_and_baseline` (`grep -n "def test_consults_ml_and_baseline" eval/agents/anomaly_detector/test_trigger.py`) y añadir el mismo decorador justo encima de su `def`:

```python
    @pytest.mark.flaky(reruns=1, reruns_delay=5)
    def test_consults_ml_and_baseline(self, ...):
```

(mantener la firma exacta existente del método — solo añadir el decorador encima; si el fichero no importa `pytest` a nivel de módulo, verificar con `grep -n "^import pytest" eval/agents/anomaly_detector/test_trigger.py` antes de usar `@pytest.mark...`).

- [ ] **Step 4: Documentar en eval/README.md**

Añadir al final de la sección "Estado de la validación live" (después del último párrafo del fichero):

```markdown

**Mitigación de flakiness LLM residual (2026-07-04):** `test_low_margin_scenario_triggers_mitigation`,
`test_ample_margin_scenario_skips_mitigation` (cross_border) y `test_consults_ml_and_baseline`
(anomaly_detector) llevan `@pytest.mark.flaky(reruns=1, reruns_delay=5)` — no-determinismo
LLM de subagentes individuales, NO infraestructura (ya eliminada, ver arriba). Los tests
del orchestrator NO llevan este marcador: tras el `escalation_planner` determinista
(spec `2026-07-04-escalation-planner-design.md`) su resultado debe ser estable.
```

- [ ] **Step 5: Suite offline completa (los marcadores flaky no afectan a offline)**

Run: `cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && PYTHONUTF8=1 python -m pytest -q`
Expected: 278 passed, 16 deselected

- [ ] **Step 6: Commit**

```bash
git add requirements.txt eval/agents/cross_border/test_trigger.py eval/agents/anomaly_detector/test_trigger.py eval/README.md
git commit -m "test(eval): pytest-rerunfailures (1 retry) en los 3 tests live con flakiness LLM conocida

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Validación live compartida (planner + lecciones)

**Nota:** esta task se ejecuta DESPUÉS de que el plan de lecciones (`docs/superpowers/plans/2026-07-04-lessons-memory.md`) también esté implementado — es la corrida live COMPARTIDA de ambas specs, tal como decidió el usuario. No dispares esta task hasta confirmar que ese otro plan llegó a su Task 4 (wiring del prompt).

**Files:**
- Modify: `eval/README.md` (resultado final)
- Modify: `.superpowers/sdd/progress.md` (ledger, no se commitea — está gitignored)

**Interfaces:**
- Consumes: `escalation_planner` wireado (Task 2-3 de este plan) + `store_lesson`/`get_lessons` wireados (plan de lecciones).
- Produces: evidencia medida del estado 12/12 (o el que sea) para cerrar honestamente la spec.

- [ ] **Step 1: Corrida live única**

Run (timeout 20 min):

```bash
cd "C:\Users\Cristian Valverde\OneDrive\Escritorio\Cursos\5dgai - GooglexKaggle\Capstone project" && PYTHONUTF8=1 python -m pytest eval/agents -m integration -v > ".superpowers/sdd/eval_live_$(date +%Y%m%d_%H%M)_planner_lessons.log" 2>&1
```

- [ ] **Step 2: Analizar el log**

```bash
grep -E "PASSED|FAILED|RERUN" .superpowers/sdd/eval_live_*_planner_lessons.log
grep -c "Timed out" .superpowers/sdd/eval_live_*_planner_lessons.log   # esperado: 0
```

Verificar específicamente: `test_run_scenario_critical` y
`test_critical_scenario_triggers_full_escalation` PASSED (la trayectoria debe
contener `escalation_planner` y las skills que su plan mande); `test_normal_scenario_triggers_nothing`
PASSED (planner VERDE → `mandatory_skills=[]`).

- [ ] **Step 3: Documentar el resultado real en `eval/README.md`**

Reemplazar la sección "Estado de la validación live" con los datos medidos del
Step 2 (no inventar cifras). Si algún test del orchestrator sigue en rojo pese
al planner determinista, documentarlo tal cual e invocar
`superpowers:systematic-debugging` — no parchear el prompt a ciegas.

- [ ] **Step 4: Commit y push**

```bash
git add eval/README.md
git commit -m "docs(eval): resultado de la validacion live con escalation_planner + lecciones

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push
```
