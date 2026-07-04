# Spec: escalación determinista (escalation_planner) + mitigación de flips LLM

**Fecha:** 2026-07-04 · **Estado:** aprobada por el usuario (opción "Planner + rerun")
**Contexto:** corrida live 2026-07-04 post-fixes de infra: 7/12, 0 timeouts. El
orchestrator detiene la escalación tras `anomaly_detector_skill` — la política de
escalación vive en prosa en `prompts/orchestrator.md` (PASO 2) y el LLM la sigue
a medias. Afecta a `test_run_scenario_critical`,
`test_critical_scenario_triggers_full_escalation` y al assert de RAG (solo se
satisface si la escalación llega a `report_agent`/`cross_border_agent`).

## Decisión

**Shift-left (Python puro > ML > LLM):** la decisión de QUÉ skills invocar es una
función determinista de (P(blackout), estado Markov). Se extrae del prompt a un
Function Tool `escalation_planner`; el prompt queda reducido a "ejecuta TODO lo
que devuelva el planner". Para el no-determinismo residual de los SUBAGENTES
(flips de cross_border/anomaly que no dependen del planner), `pytest-rerunfailures`
con 1 retry SOLO en esos tests live.

Alternativa descartada: prompt más directivo sin código — sigue dependiendo de la
obediencia del LLM, que es exactamente lo que falla.

## Diseño

### 1. Function Tool `tools/escalation_planner.py`

```python
def escalation_planner(blackout_probability: float, markov_state: int) -> dict
```

Devuelve un plan cerrado:

```python
{
  "band": "VERDE" | "AMARILLO" | "NARANJA" | "ROJO",
  "mandatory_skills": [...],       # nombres runtime EXACTOS, en orden de invocación
  "cross_border_mode": "none" | "assessment" | "emergency",
  "invoke_report_agent": bool,     # True solo en ROJO (report_type="emergency")
  "emit_alert": "none" | "warning" | "critical",
  "scope": "peninsular" | "nacional",
  "reason": str,                   # explica overrides (p.ej. "Markov S2 eleva VERDE→AMARILLO")
}
```

Reglas (transcripción 1:1 del PASO 2 actual del prompt — la fuente de verdad pasa
a ser el código):
- Bandas por P: VERDE `<0.10`, AMARILLO `[0.10, 0.30)`, NARANJA `[0.30, 0.60)`, ROJO `>=0.60`.
- Override 1: Markov `>= 2` (S2+) con P `< 0.10` → eleva a AMARILLO.
- Override 2: Markov `== 1` (S1) con P en `[0.10, 0.30)` → añade cross_border
  `assessment` al flujo AMARILLO.
- `mandatory_skills` por banda (nombres runtime): VERDE → `[]`; AMARILLO →
  `["anomaly_detector_skill"]`; NARANJA → `["anomaly_detector_skill",
  "renewables_forecast_skill", "regulatory_compliance_skill"]` +
  cross_border assessment; ROJO → las 3 skills + cross_border `emergency` +
  `invoke_report_agent=True`.
- Alertas: NARANJA → `warning`/`peninsular`; ROJO → `critical`/`nacional`.
- Validación de entrada: P se recorta a [0.0, 1.0]; markov_state a [0, 4]
  (valores fuera de rango no lanzan: se recortan y se anota en `reason`).

### 2. Prompt `prompts/orchestrator.md`

- PASO 1 gana el sub-paso: tras `risk_scorer`, llamar a `escalation_planner` con
  el `blackout_probability` del ensemble y el estado numérico de `get_grid_state`.
- El PASO 2 en prosa (bandas, overrides, listas por tramo) se REEMPLAZA por:
  ejecuta TODAS las tools de `mandatory_skills` en orden, sin saltar ninguna
  aunque creas que no aportan; invoca `cross_border_agent` según
  `cross_border_mode`; `alert_formatter` + alerta según `emit_alert`/`scope`;
  `report_agent` si `invoke_report_agent`. El campo `actions_taken` del JSON
  final debe listar lo ejecutado.
- Las REGLAS NO NEGOCIABLES y el FORMATO DE RESPUESTA no cambian (el fallback
  "ML caído → P=0.20" sigue: con P=0.20 el planner devuelve AMARILLO, coherente).

### 3. Wiring

- `agents/orchestrator.py`: importar y añadir `escalation_planner` a la lista
  `tools=[]` (capa 1, junto a `risk_scorer`).
- `guardrails/hooks.py`: añadir `"escalation_planner"` a
  `_ALLOWED_TOOLS["gridguard_orchestrator"]`.

### 4. pytest-rerunfailures (solo subagentes, solo live)

- Añadir `pytest-rerunfailures>=14.0` a `requirements.txt` (junto a pytest).
- Marcar con `@pytest.mark.flaky(reruns=1, reruns_delay=5)` ÚNICAMENTE los tests
  live de subagentes con no-determinismo observado:
  `eval/agents/cross_border/test_trigger.py` (los 2) y
  `eval/agents/anomaly_detector/test_trigger.py::test_consults_ml_and_baseline`.
- NO marcar los tests del orchestrator: deben reflejar la verdad del planner
  sin maquillaje.
- Documentar en `eval/README.md` que los reruns mitigan no-determinismo LLM de
  subagentes, no infraestructura (ya eliminada).

### 5. Testing

1. Unit tests exhaustivos del planner en `tests/tools/test_escalation_planner.py`:
   tabla de verdad de las 4 bandas × estados Markov relevantes (S0, S1, S2, S4),
   los 2 overrides, bordes exactos (P=0.10, 0.30, 0.60), recorte de entradas
   fuera de rango.
2. Wiring: whitelist contiene `escalation_planner`; el prompt lo referencia y ya
   no contiene los tramos en prosa (asserts de texto sobre `prompts/orchestrator.md`).
3. Suite offline completa verde (261 actuales + nuevos).
4. Validación live: 1 corrida `pytest eval/agents -m integration` (~10 min de
   cuota) tras integrar TAMBIÉN la spec de lecciones (una sola corrida para ambas).

## Criterio de éxito

- `test_run_scenario_critical` y `test_critical_scenario_triggers_full_escalation`
  en verde (la trayectoria contiene las 3 skills + cross_border + report_agent).
- `test_normal_scenario_triggers_nothing` verde (planner VERDE → sin skills).
- Objetivo 12/12; los flips residuales de subagentes cubiertos por 1 rerun.
