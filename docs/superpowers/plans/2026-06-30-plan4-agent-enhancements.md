# Plan 4 — Mejoras de la capa de agentes + hooks

> **Para trabajadores agénticos:** SUB-SKILL REQUERIDO: `superpowers:subagent-driven-development` o
> `superpowers:executing-plans`. Lee el [índice maestro](2026-06-30-gridguard-incident-markov-agents-INDEX.md).
> Requiere los [Plan 2](2026-06-30-plan2-markov-layer.md) y [Plan 3](2026-06-30-plan3-mcp-extensions.md)
> completos (los agentes consumen `markov_server`, `get_weather_forecast_24h`, `get_eu_market_prices`,
> `get_state_transitions`).

**Objetivo:** elevar la capa de agentes de "enrutar y formatear" a "razonar": clasificación causal de
anomalías, cumplimiento normativo, previsión a 24 h, post-mortems, contexto de mercado europeo y escalación
guiada por la trayectoria de Markov además de P(blackout).

**Arquitectura:** un Skill nuevo (`RegulatoryComplianceSkill`), extensiones de instrucción y toolset a los
5 agentes existentes, dos hooks de transición de estado, y la actualización de la whitelist de tools por
agente. **Criterio rector del handoff:** se añade razonamiento al LLM solo donde una regla determinista
perdería calidad; el estado, la trayectoria y los umbrales siguen siendo Python puro (Planes 2–3).

**Tech stack:** google-adk (`Agent`, `AgentTool`, `MCPToolset`, `StdioServerParameters`), pydantic, pytest.

## Constraints globales

Ver [índice maestro](2026-06-30-gridguard-incident-markov-agents-INDEX.md). Para este plan: **nunca hardcodear
modelos Gemini** — todo agente nuevo usa `get_model("fast_agent")` (`config/model_loader.py:29`); mínimo
privilegio en `_ALLOWED_TOOLS` (cada tool nuevo debe declararse o el guardrail lo bloqueará).

## Estrategia de test para agentes LLM

El comportamiento del LLM no se testea con pytest. Lo que SÍ se testea y es la red de seguridad de este plan:
1. **Schemas** (Pydantic) de los nuevos outputs — validación estructural.
2. **Hooks** — lógica pura determinista.
3. **`_ALLOWED_TOOLS`** — que cada tool nuevo esté autorizado para el agente correcto.
4. **Contrato de instrucción** — que el prompt contenga las directivas clave nuevas (substring assertions).
5. **Smoke de import** — que cada módulo de agente importe sin error tras añadir toolsets.

---

## Estructura de ficheros

- **Create** `agents/regulatory_compliance.py` — nuevo Skill.
- **Modify** `schemas/models.py` — `ComplianceReport`; campos nuevos en `AnomalyReport`, `RenewablesForecast`.
- **Modify** `guardrails/hooks.py` — `on_state_transition`, `on_forecast_risk_horizon`, `_ALLOWED_TOOLS`.
- **Modify** `agents/anomaly_detector.py`, `agents/renewables_forecast.py`, `agents/report_agent.py`,
  `agents/cross_border.py`, `agents/orchestrator.py` — instrucción + toolsets.
- **Create** tests en `tests/agents/` y `tests/guardrails/`.

---

### Task 1: `ComplianceReport` schema + `RegulatoryComplianceSkill`

**Files:**
- Modify: `schemas/models.py`
- Create: `agents/regulatory_compliance.py`
- Test: `tests/agents/test_regulatory_compliance.py`

**Interfaces:**
- Produce: `ComplianceReport` (pydantic); `RegulatoryComplianceSkill` (`AgentTool`).
- Consume: `search_regulations` (knowledge-base MCP), `get_grid_state` (markov MCP, Plan 3).

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/agents/test_regulatory_compliance.py
from schemas.models import ComplianceReport


def test_compliance_report_schema_valid():
    r = ComplianceReport(
        applicable_regulations=[{"id": "PO 7.4", "section": "§3.2", "trigger": "renovables >75%"}],
        mandatory_actions=["verificar reserva primaria"],
        compliance_status="warning",
        reasoning="alta penetración renovable",
    )
    assert r.compliance_status == "warning"


def test_compliance_status_rejects_invalid():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ComplianceReport(applicable_regulations=[], mandatory_actions=[],
                         compliance_status="invalido", reasoning="x")


def test_skill_module_imports():
    import agents.regulatory_compliance as rc
    assert rc.RegulatoryComplianceSkill is not None
    assert "search_regulations" in rc._INSTRUCTION
    assert "NUNCA" in rc._INSTRUCTION or "Only cite" in rc._INSTRUCTION  # no inventar procedimientos
```

- [ ] **Step 2: Ejecutar el test para verificar que falla**

Run: `python -m pytest tests/agents/test_regulatory_compliance.py -v`
Expected: FAIL (`ComplianceReport` no existe).

- [ ] **Step 3: Implementar el schema y el Skill**

En `schemas/models.py` (sección Skills, tras `RenewablesForecast`, línea ~381):

```python
class ComplianceReport(BaseModel):
    """Output estructurado de RegulatoryComplianceSkill."""
    applicable_regulations: list[dict] = Field(
        default_factory=list,
        description="[{id, section, trigger}] de procedimientos REE/ENTSO-E aplicables",
    )
    mandatory_actions: list[str] = Field(default_factory=list)
    compliance_status: Literal["compliant", "warning", "violation"]
    reasoning: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
```

Crear `agents/regulatory_compliance.py` (sigue el patrón de `anomaly_detector.py`):

```python
"""
agents/regulatory_compliance.py — Skill: RegulatoryComplianceSkill (AgentTool).

Tipo: AgentTool (síncrono, sin estado). Modelo: fast_agent, temperature 0.1.
Trigger (orquestador): P≥0.30 O estado de Markov ≥ S2.
Determina qué procedimientos operativos REE/ENTSO-E aplican al estado actual de la red.
"""

from __future__ import annotations

from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

from config.model_loader import get_model
from schemas.models import ComplianceReport

_INSTRUCTION = """
Eres el agente de cumplimiento normativo de GridGuard España. Dado el estado actual de la
red, determinas qué procedimientos operativos (PO) de REE/ENTSO-E aplican y si la situación
dispara acciones obligatorias.

FLUJO:
1. Recibes un snapshot (renewables_pct, demand_mw, balance_mw, grid_state, anomalías).
2. Llama a `get_grid_state` para confirmar el estado de Markov y su trayectoria.
3. Llama a `search_regulations` con una query construida desde el estado actual, p.ej.:
     - renovables altas  → "procedimiento operativo frecuencia inercia renovables"
     - reserva activando  → "reserva secundaria activación umbral"
     - estado ≥ S3        → "importación emergencia interconexión"
4. Para cada regulación encontrada, evalúa si el estado actual la dispara.
5. Devuelve un ComplianceReport.

REGLAS:
- NUNCA calcules umbrales numéricos tú mismo — vienen en el snapshot.
- Cita SOLO procedimientos devueltos por `search_regulations`; NUNCA inventes IDs de PO.
- Si no hay regulaciones aplicables, devuelve compliance_status="compliant".

Devuelve ÚNICAMENTE el ComplianceReport en JSON estructurado.
""".strip()

_kb_toolset = MCPToolset(connection_params=StdioServerParameters(
    command="python", args=["-m", "mcp_servers.knowledge_base_server"]))
_markov_toolset = MCPToolset(connection_params=StdioServerParameters(
    command="python", args=["-m", "mcp_servers.markov_server"]))

_inner_agent = Agent(
    model=get_model("fast_agent"),
    name="regulatory_compliance_skill",
    instruction=_INSTRUCTION,
    tools=[_kb_toolset, _markov_toolset],
    output_schema=ComplianceReport,
    temperature=0.1,
)

RegulatoryComplianceSkill = AgentTool(agent=_inner_agent)
```

- [ ] **Step 4: Ejecutar los tests + commit**

Run: `python -m pytest tests/agents/test_regulatory_compliance.py -v`
Expected: PASS (3 tests).

```bash
git add schemas/models.py agents/regulatory_compliance.py tests/agents/test_regulatory_compliance.py
git commit -m "feat(agents): RegulatoryComplianceSkill + ComplianceReport schema"
```

---

### Task 2: Hooks de transición de estado

**Files:**
- Modify: `guardrails/hooks.py`
- Test: `tests/guardrails/test_state_hooks.py`

**Interfaces:**
- Produce:
  - `on_state_transition(prev_state: int, curr_state: int) -> dict` → `{"action": str, "trigger": str|None}`.
    Si `prev_state ≥ S2` y `curr_state == S0` → `action="post_mortem"`. Si `prev_state == S0` y
    `curr_state ≥ S2` → `action="regulatory_check"`. Si no → `action="none"`.
  - `on_forecast_risk_horizon(window_label: str, severity: str) -> dict` → registra y devuelve
    `{"logged": True, "advance_warning": window_label}`.

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/guardrails/test_state_hooks.py
from guardrails import hooks


def test_recovery_to_normal_triggers_post_mortem():
    assert hooks.on_state_transition(prev_state=3, curr_state=0)["action"] == "post_mortem"


def test_onset_from_normal_triggers_regulatory_check():
    assert hooks.on_state_transition(prev_state=0, curr_state=2)["action"] == "regulatory_check"


def test_minor_change_is_noop():
    assert hooks.on_state_transition(prev_state=0, curr_state=1)["action"] == "none"


def test_forecast_risk_horizon_logs_window():
    out = hooks.on_forecast_risk_horizon("14:00-14:30 UTC", "HIGH")
    assert out["logged"] is True
    assert out["advance_warning"] == "14:00-14:30 UTC"
```

- [ ] **Step 2: Ejecutar el test para verificar que falla**

Run: `python -m pytest tests/guardrails/test_state_hooks.py -v`
Expected: FAIL (`on_state_transition` no existe).

- [ ] **Step 3: Implementar los hooks**

```python
# guardrails/hooks.py  (añadir; usa el `log` ya definido en el módulo)

_S2_NOTABLE = 2


def on_state_transition(prev_state: int, curr_state: int) -> dict:
    """Reacción a un cambio de estado de Markov entre ciclos (ver Plan 4).

    - S2+ → S0 (recuperación): dispara post-mortem.
    - S0 → S2+ (onset abrupto): dispara verificación normativa inmediata.
    """
    if prev_state >= _S2_NOTABLE and curr_state == 0:
        log.info("[HOOK] on_state_transition: recuperación S%d→S0 → post_mortem", prev_state)
        return {"action": "post_mortem", "trigger": f"S{prev_state}->S0"}
    if prev_state == 0 and curr_state >= _S2_NOTABLE:
        log.warning("[HOOK] on_state_transition: onset S0→S%d → regulatory_check", curr_state)
        return {"action": "regulatory_check", "trigger": f"S0->S{curr_state}"}
    return {"action": "none", "trigger": None}


def on_forecast_risk_horizon(window_label: str, severity: str) -> dict:
    """Registra una ventana de riesgo 6–24 h detectada por RenewablesForecastSkill."""
    log.warning("[HOOK] on_forecast_risk_horizon: ventana de riesgo %s (%s)", window_label, severity)
    return {"logged": True, "advance_warning": window_label}
```

- [ ] **Step 4: Ejecutar los tests + commit**

Run: `python -m pytest tests/guardrails/test_state_hooks.py -v`
Expected: PASS (4 tests).

```bash
git add guardrails/hooks.py tests/guardrails/test_state_hooks.py
git commit -m "feat(guardrails): hooks on_state_transition y on_forecast_risk_horizon"
```

---

### Task 3: Actualizar `_ALLOWED_TOOLS` para los tools nuevos

**Files:**
- Modify: `guardrails/hooks.py` (`_ALLOWED_TOOLS`, líneas 57–81)
- Test: `tests/guardrails/test_allowed_tools.py`

**Interfaces:**
- Produce: whitelist ampliada. Sin esto, `before_tool_callback` (`hooks.py:147`) bloqueará los tools nuevos.

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/guardrails/test_allowed_tools.py
from guardrails.hooks import _ALLOWED_TOOLS


def test_orchestrator_can_use_grid_state_and_compliance():
    o = _ALLOWED_TOOLS["gridguard_orchestrator"]
    assert "get_grid_state" in o
    assert "RegulatoryComplianceSkill" in o


def test_anomaly_can_use_markov_and_weather():
    a = _ALLOWED_TOOLS["anomaly_detector_skill"]
    assert {"get_grid_state", "get_wind_forecast", "get_solar_radiation"} <= a


def test_renewables_can_use_24h_forecast():
    assert "get_weather_forecast_24h" in _ALLOWED_TOOLS["renewables_forecast_skill"]


def test_report_can_use_trajectory_and_transitions():
    r = _ALLOWED_TOOLS["report_agent"]
    assert {"get_markov_trajectory", "get_state_transitions"} <= r


def test_cross_border_can_use_market_prices():
    assert "get_eu_market_prices" in _ALLOWED_TOOLS["cross_border_agent"]


def test_regulatory_compliance_whitelist_exists():
    rc = _ALLOWED_TOOLS["regulatory_compliance_skill"]
    assert {"search_regulations", "get_grid_state"} <= rc
```

- [ ] **Step 2: Ejecutar el test para verificar que falla**

Run: `python -m pytest tests/guardrails/test_allowed_tools.py -v`
Expected: FAIL (tools nuevos aún no listados).

- [ ] **Step 3: Ampliar `_ALLOWED_TOOLS`**

```python
# guardrails/hooks.py  — reemplazar el dict _ALLOWED_TOOLS por:
_ALLOWED_TOOLS: dict[str, set[str]] = {
    "gridguard_orchestrator": {
        "ree_poller", "data_normalizer", "stability_proxy_calculator",
        "risk_scorer", "alert_formatter",
        "get_blackout_probability", "store_snapshot", "store_risk_score",
        "AnomalyDetectorSkill", "RenewablesForecastSkill",
        "CrossBorderAgent", "ReportAgent",
        "get_grid_state", "store_grid_state", "RegulatoryComplianceSkill",  # nuevos (Plan 4)
    },
    "anomaly_detector_skill": {
        "get_anomaly_score", "stability_proxy_calculator",
        "get_baseline", "search_regulations",
        "get_grid_state", "get_wind_forecast", "get_solar_radiation",  # nuevos
    },
    "renewables_forecast_skill": {
        "get_wind_forecast", "get_solar_radiation", "get_generation_mix",
        "get_load_forecast", "get_baseline",
        "get_weather_forecast_24h", "get_grid_state",  # nuevos
    },
    "report_agent": {
        "query_incidents", "get_risk_history", "get_blackout_probability",
        "find_similar_incident", "write_report",
        "get_markov_trajectory", "get_state_transitions",  # nuevos
    },
    "cross_border_agent": {
        "get_cross_border_flows", "get_capacity_margin",
        "get_eu_generation", "get_mitigation_procedure",
        "get_eu_market_prices",  # nuevo
    },
    "regulatory_compliance_skill": {  # agente nuevo (Plan 4)
        "search_regulations", "get_grid_state",
    },
}
```

- [ ] **Step 4: Ejecutar los tests + commit**

Run: `python -m pytest tests/guardrails/test_allowed_tools.py -v`
Expected: PASS (6 tests).

```bash
git add guardrails/hooks.py tests/guardrails/test_allowed_tools.py
git commit -m "feat(guardrails): whitelist de tools ampliada para markov, weather24h y compliance"
```

---

### Task 4: Extender `AnomalyDetectorSkill` (razonamiento causal + clasificación de incidente)

**Files:**
- Modify: `schemas/models.py` (`AnomalyReport`: campos `incident_type`, `causal_chain`)
- Modify: `agents/anomaly_detector.py`
- Test: `tests/agents/test_anomaly_detector_ext.py`

**Interfaces:**
- Produce: `AnomalyReport` con `incident_type: Literal["renewable_ramp","demand_surge","generation_failure","interconnection","unknown"]`
  y `causal_chain: str`. Toolset con `markov_server` y weather.

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/agents/test_anomaly_detector_ext.py
from schemas.models import AnomalyReport


def test_anomaly_report_has_incident_type_and_causal_chain():
    r = AnomalyReport(
        anomalies=[], severity="MEDIUM", confidence=0.5, ml_score=0.4,
        regulatory_context="", reasoning="x",
        incident_type="renewable_ramp", causal_chain="subida solar brusca → balance positivo",
    )
    assert r.incident_type == "renewable_ramp"


def test_instruction_mentions_incident_classification():
    import agents.anomaly_detector as ad
    assert "incident_type" in ad._INSTRUCTION
    assert "get_grid_state" in ad._INSTRUCTION


def test_module_imports():
    import agents.anomaly_detector as ad
    assert ad.AnomalyDetectorSkill is not None
```

- [ ] **Step 2: Ejecutar el test para verificar que falla**

Run: `python -m pytest tests/agents/test_anomaly_detector_ext.py -v`
Expected: FAIL (`incident_type` no existe en `AnomalyReport`).

- [ ] **Step 3: Añadir campos al schema, extender instrucción y toolset**

En `schemas/models.py`, `AnomalyReport` (línea ~353), añadir antes de `timestamp`:

```python
    incident_type: Literal[
        "renewable_ramp", "demand_surge", "generation_failure", "interconnection", "unknown"
    ] = "unknown"
    causal_chain: str = Field(default="", description="Cadena causal que explica la anomalía")
```

En `agents/anomaly_detector.py`, añadir al `_INSTRUCTION` tras el paso 4 (línea ~38) y antes de "REGLAS":

```
4b. Llama a `get_grid_state` (MCP markov) para obtener el estado actual S0–S4 y su trayectoria 6h.
4c. Llama a `get_wind_forecast` y `get_solar_radiation` en las zonas más correlacionadas con la anomalía.
5b. Razona explícitamente qué combinación de factores (rampa de generación, meteo, nivel de demanda,
    trayectoria de estado) explica mejor el score del ML, y CLASIFICA el incidente en incident_type:
      - "renewable_ramp"     : cambio brusco de solar/eólica
      - "demand_surge"       : pico de demanda inesperado
      - "generation_failure" : caída de generación convencional
      - "interconnection"    : anomalía de flujo transfronterizo
      - "unknown"            : evidencia insuficiente
    Rellena `incident_type` y `causal_chain` en el AnomalyReport.
```

Añadir los toolsets markov + weather y registrarlos en `tools=[...]` (línea ~82):

```python
_markov_toolset = MCPToolset(connection_params=StdioServerParameters(
    command="python", args=["-m", "mcp_servers.markov_server"]))
_weather_toolset = MCPToolset(connection_params=StdioServerParameters(
    command="python", args=["-m", "mcp_servers.weather_server"]))
# tools=[stability_proxy_calculator, _ml_toolset, _memory_toolset, _kb_toolset,
#        _markov_toolset, _weather_toolset]
```

- [ ] **Step 4: Ejecutar los tests + commit**

Run: `python -m pytest tests/agents/test_anomaly_detector_ext.py -v`
Expected: PASS (3 tests).

```bash
git add schemas/models.py agents/anomaly_detector.py tests/agents/test_anomaly_detector_ext.py
git commit -m "feat(agents): anomaly_detector clasifica incident_type y razona causalidad con markov+meteo"
```

---

### Task 5: Extender `RenewablesForecastSkill` (horizonte 24 h + Markov)

**Files:**
- Modify: `agents/renewables_forecast.py`
- Test: `tests/agents/test_renewables_forecast_ext.py`

**Interfaces:**
- Consume: `get_weather_forecast_24h` (Plan 3), `get_grid_state`.
- Produce: instrucción extendida (ventana 6–24 h) + toolset weather24h/markov. (El schema `RenewablesForecast`
  ya tiene `critical_window` y `reasoning`; no requiere campo nuevo.)

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/agents/test_renewables_forecast_ext.py
def test_instruction_mentions_24h_and_markov():
    import agents.renewables_forecast as rf
    assert "get_weather_forecast_24h" in rf._INSTRUCTION
    assert "get_grid_state" in rf._INSTRUCTION


def test_module_imports():
    import agents.renewables_forecast as rf
    assert rf.RenewablesForecastSkill is not None
```

- [ ] **Step 2: Ejecutar el test para verificar que falla**

Run: `python -m pytest tests/agents/test_renewables_forecast_ext.py -v`
Expected: FAIL.

- [ ] **Step 3: Extender instrucción y toolset**

En `agents/renewables_forecast.py`, añadir al `_INSTRUCTION` tras el paso de `get_solar_radiation`:

```
4b. Llama a `get_weather_forecast_24h` para todas las zonas y extiende el horizonte más allá de 6h.
6b. Identifica no solo la ventana crítica de 6h, sino también eventos de rampa de alto riesgo en 6–24h
    (p.ej. "tormenta a las 14:00 de mañana reducirá la solar en 6 GW en <30 min — recomendar activar
    reserva antes de las 13:30").
7b. Llama a `get_grid_state` para incluir la trayectoria de Markov actual en la evaluación de riesgo.
```

Añadir toolsets weather24h (el mismo `weather_server` ya expone el tool) y markov, y registrarlos en
`tools=[...]`:

```python
_markov_toolset = MCPToolset(connection_params=StdioServerParameters(
    command="python", args=["-m", "mcp_servers.markov_server"]))
# Asegúrate de que el weather_toolset del módulo incluye get_weather_forecast_24h (mismo server)
# y de añadir _markov_toolset a la lista tools=[...] del Agent.
```

- [ ] **Step 4: Ejecutar los tests + commit**

Run: `python -m pytest tests/agents/test_renewables_forecast_ext.py -v`
Expected: PASS.

```bash
git add agents/renewables_forecast.py tests/agents/test_renewables_forecast_ext.py
git commit -m "feat(agents): renewables_forecast a 24h con contexto de trayectoria Markov"
```

---

### Task 6: Extender `ReportAgent` (tipo `post_mortem` + trayectoria)

**Files:**
- Modify: `agents/report_agent.py`
- Test: `tests/agents/test_report_agent_ext.py`

**Interfaces:**
- Consume: `get_markov_trajectory`, `get_state_transitions` (Plan 3).
- Produce: instrucción con el nuevo `report_type="post_mortem"` y la llamada a trayectoria.

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/agents/test_report_agent_ext.py
def test_instruction_supports_post_mortem():
    import agents.report_agent as ra
    assert "post_mortem" in ra._INSTRUCTION
    assert "get_state_transitions" in ra._INSTRUCTION
    assert "get_markov_trajectory" in ra._INSTRUCTION


def test_module_imports():
    import agents.report_agent as ra
    assert ra.report_agent is not None
```

- [ ] **Step 2: Ejecutar el test para verificar que falla**

Run: `python -m pytest tests/agents/test_report_agent_ext.py -v`
Expected: FAIL.

- [ ] **Step 3: Extender instrucción y toolset**

En `agents/report_agent.py`, añadir al `_INSTRUCTION` el nuevo tipo de informe:

```
Tipo de informe "post_mortem" (NUEVO):
  Se dispara cuando el estado de Markov vuelve a S0 tras un episodio S2+.
  1. `get_state_transitions` — recupera la secuencia completa de estados del episodio.
  2. `get_risk_history` con la ventana temporal del episodio.
  3. `find_similar_incident` — compara con incidentes históricos del RAG.
  4. Sintetiza: señales de onset, trayectoria de estado, duración, causa probable, resolución.
  5. `write_report` con tipo post_mortem.

Para los tipos de informe existentes:
  - Añade una llamada a `get_markov_trajectory` tras get_risk_history (muestra la senda S0→S1→S2 en el informe).
```

Añadir toolsets markov (`markov_server`) y, si no lo tiene ya, el memory toolset que expone
`get_state_transitions`; registrarlos en `tools=[...]`.

- [ ] **Step 4: Ejecutar los tests + commit**

Run: `python -m pytest tests/agents/test_report_agent_ext.py -v`
Expected: PASS.

```bash
git add agents/report_agent.py tests/agents/test_report_agent_ext.py
git commit -m "feat(agents): report_agent con post_mortem y trayectoria de estado"
```

---

### Task 7: Extender `CrossBorderAgent` (parámetro `mode` + precios de mercado)

**Files:**
- Modify: `agents/cross_border.py`
- Test: `tests/agents/test_cross_border_ext.py`

**Interfaces:**
- Consume: `get_eu_market_prices` (Plan 3).
- Produce: instrucción con `mode` ∈ {assessment, emergency} + uso de precios (spread ES-FR > 20 €/MWh →
  probable excedente en Francia).

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/agents/test_cross_border_ext.py
def test_instruction_supports_mode_and_prices():
    import agents.cross_border as cb
    assert "assessment" in cb._INSTRUCTION
    assert "emergency" in cb._INSTRUCTION
    assert "get_eu_market_prices" in cb._INSTRUCTION


def test_module_imports():
    import agents.cross_border as cb
    assert cb.cross_border_agent is not None
```

- [ ] **Step 2: Ejecutar el test para verificar que falla**

Run: `python -m pytest tests/agents/test_cross_border_ext.py -v`
Expected: FAIL.

- [ ] **Step 3: Extender instrucción y toolset**

En `agents/cross_border.py`, añadir al `_INSTRUCTION`:

```
PARÁMETRO `mode` (lo pasa el orquestador):
  mode="assessment": evalúa proactivamente la capacidad de importación disponible; sin urgencia (P∈[0.30,0.60)).
  mode="emergency" : flujo actual urgente (P>0.60).

Paso adicional (tras get_eu_generation):
3b. Llama a `get_eu_market_prices` para ver si los países vecinos tienen excedente barato
    (precios bajos → probable excedente de generación → importación más fácil).
    Nota: si el spread de precio ES-FR > 20 €/MWh, Francia probablemente tiene excedente.
```

Añadir toolset que exponga `get_eu_market_prices` (el mismo `entsoe_server`, ya conectado en este módulo —
basta con que el tool esté disponible) y verificar que `get_eu_market_prices` quede dentro del toolset.

- [ ] **Step 4: Ejecutar los tests + commit**

Run: `python -m pytest tests/agents/test_cross_border_ext.py -v`
Expected: PASS.

```bash
git add agents/cross_border.py tests/agents/test_cross_border_ext.py
git commit -m "feat(agents): cross_border con modo assessment/emergency y precios de mercado EU"
```

---

### Task 8: Extender `OrchestratorAgent` (estado de Markov + nuevos umbrales)

**Files:**
- Modify: `agents/orchestrator.py`
- Test: `tests/agents/test_orchestrator_ext.py`

**Interfaces:**
- Consume: `get_grid_state` (Plan 3), `RegulatoryComplianceSkill` (Task 1).
- Produce: instrucción con adquisición de estado de Markov (paso 1d), umbrales que combinan P(blackout) y
  estado, e invocación de `RegulatoryComplianceSkill` y `CrossBorderAgent(mode="assessment")` en NARANJA.

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/agents/test_orchestrator_ext.py
def test_instruction_has_markov_and_new_thresholds():
    import agents.orchestrator as orch
    text = orch._INSTRUCTION
    assert "get_grid_state" in text
    assert "RegulatoryComplianceSkill" in text
    assert "assessment" in text  # CrossBorder en modo assessment en NARANJA
    # Override Markov: estado ≥ S2 con P baja sube a AMARILLO
    assert "S2" in text


def test_module_imports():
    import agents.orchestrator as orch
    assert orch is not None
```

- [ ] **Step 2: Ejecutar el test para verificar que falla**

Run: `python -m pytest tests/agents/test_orchestrator_ext.py -v`
Expected: FAIL.

- [ ] **Step 3: Extender la instrucción, importar el Skill y cablear el toolset**

En `agents/orchestrator.py`, en PASO 1 (tras 1c, línea ~60):

```
  1d. [NUEVO] Llama a `get_grid_state` (MCP markov) para obtener el estado actual S0–S4
      y la trayectoria de las últimas 6h.
  1e. Llama a `get_blackout_probability` (como antes).
```

En PASO 2 (escalación), añadir las reglas combinadas y actualizar el tramo NARANJA:

```
La escalación usa AHORA tanto P(blackout) COMO el estado de Markov:
  - Si estado Markov ≥ S2 Y P < 0.10: eleva a AMARILLO (el estado manda cuando el ML duda
    pero la trayectoria se degrada claramente).
  - Si estado Markov = S1 Y P ∈ [0.10, 0.30): activa CrossBorderAgent en modo assessment.

▶ P ∈ [0.30, 0.60) (NARANJA) — [ACTUALIZADO]:
  → Invoca AnomalyDetectorSkill.
  → Invoca RenewablesForecastSkill.
  → [NUEVO] Invoca RegulatoryComplianceSkill.
  → [NUEVO] Invoca CrossBorderAgent con mode="assessment".
  → Llama a alert_formatter con las 4 entradas anteriores.
  → Emite alerta WARNING incluyendo la capacidad de importación disponible.
```

Importar y registrar el nuevo Skill y el toolset markov:

```python
from agents.regulatory_compliance import RegulatoryComplianceSkill  # junto a los demás imports de agents

_markov_toolset = MCPToolset(connection_params=StdioServerParameters(
    command="python", args=["-m", "mcp_servers.markov_server"]))
# Añadir RegulatoryComplianceSkill y _markov_toolset a la lista tools=[...] del orchestrator Agent.
```

- [ ] **Step 4: Ejecutar los tests + smoke de import + commit**

Run: `python -m pytest tests/agents/test_orchestrator_ext.py -v`
Expected: PASS.
Run: `python -c "import agents.orchestrator"`
Expected: sin errores de import.

```bash
git add agents/orchestrator.py tests/agents/test_orchestrator_ext.py
git commit -m "feat(agents): orchestrator escala por estado de Markov + RegulatoryComplianceSkill en NARANJA"
```

---

### Task 9: Test de integración end-to-end con el golden dataset

**Files:**
- Create/Modify: `eval/test_outputs.py` o `eval/test_trajectories.py` (seguir el marcado `-m integration`)
- Test: el propio fichero de eval

**Interfaces:**
- Consume: todo lo anterior + `eval/golden_dataset/blackout_apr_2025.json`.

- [ ] **Step 1: Escribir el test de integración**

```python
# eval/test_trajectories.py  (añadir)
import json
from pathlib import Path
import pytest


@pytest.mark.integration
def test_markov_trajectory_escalates_on_blackout_scenario():
    """Sobre los snapshots pre-apagón, P(alcanzar S4 en 6h) no debe decrecer hacia el colapso."""
    from tools import markov_state_tracker as mst
    from ml.incidents import S4_BLACKOUT
    from ml.train.validate_models import _snapshot_to_row

    scenario = json.loads(Path("eval/golden_dataset/blackout_apr_2025.json").read_text(encoding="utf-8"))
    prev_flow = None
    probs = []
    for snap in scenario["snapshots"]:
        row, prev_flow = _snapshot_to_row(snap, prev_flow)
        features = {k: row[k] for k in ("composite_pressure", "renewables_pct", "balance_mw")}
        state = mst.compute_state(features)
        probs.append(mst.prob_reach_severity(state, S4_BLACKOUT, horizon_hours=6))

    # La P de escalada en el último snapshot (pre-colapso) ≥ la del primero del escenario
    assert probs[-1] >= probs[0]
```

- [ ] **Step 2: Ejecutar la suite de integración**

Run: `pytest eval/ -m integration -v`
Expected: PASS (requiere `ml/models/markov_state.npz` del Plan 2 y los modelos del Plan 1).

- [ ] **Step 3: Demo del orquestador con el golden dataset**

Run el flujo de demo del orquestador sobre el escenario del apagón 28-abr-2025 y verifica manualmente que:
estado de Markov escala S0→S1→S2; en NARANJA se invocan RegulatoryComplianceSkill y CrossBorderAgent
(assessment); el informe final incluye la trayectoria. Captura la salida para el entregable del capstone.

- [ ] **Step 4: Commit**

```bash
git add eval/test_trajectories.py
git commit -m "test(eval): integracion de trayectoria de Markov sobre el golden dataset del apagon"
```

---

## Cierre del Plan 4

Al terminar: la capa de agentes razona sobre causalidad (`incident_type`), cumplimiento normativo
(`RegulatoryComplianceSkill`), horizonte 24 h, post-mortems y mercado europeo; el orquestador escala por
estado de Markov además de P(blackout); los hooks reaccionan a transiciones de estado; la whitelist autoriza
todos los tools nuevos. El sistema completo (Planes 1–4) pasa de "ML hace todo, agentes formatean" a una
arquitectura en capas donde cada capa aporta lo que la inferior no puede.

Run de regresión total: `python -m pytest tests/ -v && pytest eval/ -m integration -v`
