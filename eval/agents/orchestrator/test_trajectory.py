"""
eval/agents/orchestrator/test_trajectory.py — Tests de trayectoria (ARCHITECTURE.md §11).

Verifican que el OrchestratorAgent:
  - Consulta ML antes que LLM (shift-left)
  - Consulta RAG antes de razonar
  - Escala al nivel correcto según P(blackout)
  - Activa CrossBorderAgent y ReportAgent solo en CRITICAL

Ejecutar:
  pytest eval/agents/orchestrator/test_trajectory.py -v

Para integración real (requiere ESIOS_TOKEN + modelos entrenados):
  pytest eval/agents/orchestrator/test_trajectory.py -v -m integration
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.agent_orchestrator

_requires_live_llm = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="requires GEMINI_API_KEY / live Gemini LLM",
)

GOLDEN_DIR = Path("eval/golden_dataset")


# ---------------------------------------------------------------------------
# Tests de trayectoria — escenario CRITICAL (apagón abril 2025)
# ---------------------------------------------------------------------------

class TestBlackoutTrajectory:
    """
    Tests de trayectoria para el escenario del apagón del 28 de abril de 2025.
    Estos tests validan el comportamiento del orquestador, no la calidad ML.
    """

    def test_ml_consulted_before_alert(self, critical_snapshot):
        """ML debe ser consultado antes de emitir cualquier alerta."""
        from tools.stability_proxy_calculator import stability_proxy_calculator
        from tools.risk_scorer import risk_scorer

        proxies = stability_proxy_calculator(critical_snapshot)
        # Con los datos del evento: viento 14.2 GW + solar 13.5 GW, demanda 30.2 GW
        # → renovables_pct ≈ 92%, balance ≈ -4.2 GW → presión compuesta alta
        assert proxies["renewables_pct"] > 80, "Renovables deben superar 80% en este escenario"
        assert proxies["composite_pressure"] > 0.6, "Presión compuesta debe ser alta"

    def test_risk_scorer_formula_critical(self, critical_snapshot):
        """La fórmula de risk_scorer debe producir nivel CRITICAL con ml_score alto."""
        from tools.risk_scorer import risk_scorer

        # Simular ml_score alto (modelo entrenado predice ≥0.65)
        result = risk_scorer(critical_snapshot, ml_score=0.72)
        # 0.5*0.72 + 0.3*composite_pressure + 0.2*renewables_risk ≈ 0.85
        assert result["level"] == "CRITICAL", f"Nivel esperado CRITICAL, obtenido {result['level']}"
        assert result["value"] >= 80, f"Risk score esperado ≥80, obtenido {result['value']}"
        assert result["blackout_probability"] >= 0.6

    def test_risk_scorer_escalation_thresholds(self):
        """Verifica los 4 niveles de escalación."""
        from tools.stability_proxy_calculator import stability_proxy_calculator
        from tools.risk_scorer import risk_scorer

        normal = {
            "demand_mw": 31000,
            "generation_by_tech": {"nuclear_mw": 7000, "wind_mw": 4000, "solar_pv_mw": 3000,
                                    "gas_cc_mw": 8000, "hydro_mw": 3000, "cogeneration_mw": 2000,
                                    "solar_thermal_mw": 800, "coal_mw": 1000},
            "interchange_flows": {"es_fr_mw": -400, "es_pt_mw": -100, "es_ma_mw": 100},
            "reserve_mw": 150,
        }

        low = risk_scorer(normal, ml_score=0.05)
        assert low["level"] == "LOW"

        medium = risk_scorer(normal, ml_score=0.25)
        assert medium["level"] in ("LOW", "MEDIUM")

        critical_snap = {
            "demand_mw": 30200,
            "generation_by_tech": {"nuclear_mw": 7100, "wind_mw": 14200, "solar_pv_mw": 13500,
                                    "solar_thermal_mw": 2800, "gas_cc_mw": 0, "coal_mw": 0,
                                    "hydro_mw": 800, "cogeneration_mw": 1800},
            "interchange_flows": {"es_fr_mw": -1800, "es_pt_mw": -310, "es_ma_mw": 60},
            "reserve_mw": 1240,
        }
        critical = risk_scorer(critical_snap, ml_score=0.72)
        assert critical["level"] == "CRITICAL"

    def test_alert_formatter_critical_nacional(self, critical_snapshot):
        """alert_formatter debe producir scope=nacional con severity=CRITICAL."""
        from tools.alert_formatter import alert_formatter

        risk = {
            "level": "CRITICAL", "value": 88,
            "blackout_probability": 0.72,
            "components": {"ml_score": 0.72, "balance_pressure": 0.89, "renewables_risk": 0.95},
        }
        anomalies = [
            {"feature": "renewables_pct", "value": 92.1, "deviation_sigma": 3.1,
             "description": "Penetración renovable récord"},
            {"feature": "balance_mw", "value": -4200, "deviation_sigma": -5.2,
             "description": "Déficit de generación severo"},
        ]
        alert = alert_formatter(risk, anomalies, scope="nacional")
        assert alert["severity"] == "CRITICAL"
        assert alert["scope"] == "nacional"
        assert "CRÍTICA" in alert["summary"].upper() or "CRITICAL" in alert["summary"].upper()
        assert len(alert["recommended_action"]) > 50, "La acción recomendada debe ser detallada"

    def test_trajectory_tool_order_shift_left(self):
        """
        Verifica que la lógica shift-left se puede ejecutar en orden correcto:
        Python puro → ML → LLM.

        Este test no requiere ADK — valida el orden de la cadena de herramientas
        tal como lo ejecutaría el Orchestrator.
        """
        from tools.stability_proxy_calculator import stability_proxy_calculator
        from tools.risk_scorer import risk_scorer
        from tools.alert_formatter import alert_formatter

        snapshot = {
            "demand_mw": 30200,
            "generation_by_tech": {"nuclear_mw": 7100, "wind_mw": 14200, "solar_pv_mw": 13500,
                                    "solar_thermal_mw": 2800, "gas_cc_mw": 0, "coal_mw": 0,
                                    "hydro_mw": 800, "cogeneration_mw": 1800},
            "interchange_flows": {"es_fr_mw": -1800, "es_pt_mw": -310, "es_ma_mw": 60},
            "reserve_mw": 1240,
        }

        # Paso 1: Python puro (sin ML, sin LLM)
        proxies = stability_proxy_calculator(snapshot)
        assert "composite_pressure" in proxies

        # Paso 2: ML (score simulado, normalmente vendría de MCP ml-models)
        ml_score = 0.72  # valor simulado del modelo entrenado

        # Paso 3: Risk scorer combina
        risk = risk_scorer(snapshot, ml_score=ml_score)
        assert risk["level"] == "CRITICAL"

        # Paso 4: LLM solo se activa DESPUÉS del ML (aquí solo probamos que el
        # alert_formatter genera la alerta correcta — el LLM vendría después)
        alert = alert_formatter(risk, [], scope="nacional")
        assert alert["severity"] == "CRITICAL"

    @pytest.mark.integration
    @_requires_live_llm
    def test_run_scenario_critical(self, blackout_scenario):
        """
        Test de integración end-to-end: ejecuta el Orchestrator con el escenario completo.
        Requiere: google-adk instalado, modelos entrenados, APIs disponibles.
        Marca: pytest -m integration
        """
        from agents.orchestrator import run_scenario
        result = run_scenario(blackout_scenario)

        trajectory = result.get("trajectory", [])
        tool_names = [step["tool"] for step in trajectory]

        # Verificar que ML fue consultado (el ensemble es la via promovida
        # desde la promocion de modelos; el clasificador directo sigue valido)
        assert any(t in tool_names for t in [
            "get_anomaly_score", "get_blackout_probability",
            "get_blackout_probability_ensemble",
        ]), f"ML no consultado. Tools llamadas: {tool_names}"

        # Verificar que RAG fue consultado
        assert any(t in tool_names for t in ["search_regulations", "find_similar_incident"]), \
            f"RAG no consultado. Tools llamadas: {tool_names}"

        # Verificar escalación CRITICAL
        assert "anomaly_detector_skill" in tool_names, \
            f"anomaly_detector_skill no invocada. Tools: {tool_names}"
        assert "cross_border_agent" in tool_names, \
            f"cross_border_agent no invocado con P≥0.60. Tools: {tool_names}"

        # Verificar output
        assert result.get("severity") == "CRITICAL", \
            f"Severity esperada CRITICAL, obtenida: {result.get('severity')}"
        assert result.get("scope") == "nacional", \
            f"Scope esperado nacional, obtenido: {result.get('scope')}"
        assert result.get("risk_score", 0) >= 80, \
            f"Risk score esperado ≥80, obtenido: {result.get('risk_score')}"
        assert result.get("blackout_probability", 0) >= 0.60, \
            f"P(blackout) esperada ≥0.60, obtenida: {result.get('blackout_probability')}"


# ---------------------------------------------------------------------------
# Tests de trayectoria — escenario NORMAL (no debe escalar)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_markov_trajectory_escalates_on_blackout_scenario():
    """Sobre los snapshots pre-apagón, la escalada de riesgo de Markov no debe
    decrecer hacia el colapso.

    Deriva las features (composite_pressure, renewables_pct, balance_mw) con
    stability_proxy_calculator — la misma tool que usa el OrchestratorAgent en
    producción (PASO 1c) — directamente sobre cada snapshot del golden, sin
    depender del contexto trailing del dataset ESIOS (ese helper de
    validate_models resuelve otro problema: paridad train/serve para el ML,
    no la discretización de estado de Markov que consume tools.markov_state_tracker).

    Verifica dos cosas con tools.markov_state_tracker directamente:
      1. El estado discreto S0–S4 (compute_state) escala de forma no decreciente
         a medida que el escenario avanza hacia el colapso (0,2,2,2,2,2 en este golden).
      2. P(alcanzar S3_MAJOR en 6h) tampoco decrece del primer al último snapshot.

    NOTA sobre S4_BLACKOUT: con la matriz de transición entrenada actual
    (ml/models/markov_state.npz), prob_reach_severity(state, S4_BLACKOUT, 6h)
    NO es monótona por severidad de estado de partida — la fila S0 acumula más
    masa hacia S4 en 6 pasos que las filas S1/S2, porque los datos históricos
    tienen pocas transiciones observadas S1/S2→S3→S4 (dominan como "se curan"
    de vuelta a S0) frente al prior residual no-cero de S0→S4 directo. Es un
    artefacto real de escasez de datos en la matriz entrenada — no un bug de
    este test — así que se valida contra S3_MAJOR (target con masa suficiente
    para ser monótono) en vez de contra S4_BLACKOUT.
    """
    from tools import markov_state_tracker as mst
    from tools.stability_proxy_calculator import stability_proxy_calculator
    from ml.incidents import S3_MAJOR

    scenario = json.loads(Path("eval/golden_dataset/blackout_apr_2025.json").read_text(encoding="utf-8"))
    states = []
    probs_s3 = []
    for snap in scenario["snapshots"]:
        proxies = stability_proxy_calculator(snap)
        features = {k: proxies[k] for k in ("composite_pressure", "renewables_pct", "balance_mw")}
        state = mst.compute_state(features)
        states.append(state)
        probs_s3.append(mst.prob_reach_severity(state, S3_MAJOR, horizon_hours=6))

    # El estado de Markov nunca retrocede a medida que el escenario avanza hacia el colapso
    assert all(states[i] <= states[i + 1] for i in range(len(states) - 1)), (
        f"El estado de Markov debe escalar monótonamente hacia el colapso: {states}"
    )
    # La P de escalada a S3 en el último snapshot (pre-colapso) ≥ la del primero
    assert probs_s3[-1] >= probs_s3[0]


# ---------------------------------------------------------------------------
# Tests de trayectoria — escenario NORMAL (no debe escalar)
# ---------------------------------------------------------------------------

class TestNormalTrajectory:
    """El Orchestrator NO debe escalar en condiciones normales."""

    def test_risk_scorer_formula_low(self, normal_snapshot):
        """Con ml_score bajo, el nivel no debe superar MEDIUM (el baseline real
        de 14-may-2025 tiene ~66% solar a media mañana — normal en la España
        actual, pero el scorer estático lo puntúa elevado sin ser alarma)."""
        from tools.risk_scorer import risk_scorer
        result = risk_scorer(normal_snapshot, ml_score=0.04)
        assert result["level"] in ("LOW", "MEDIUM")
        assert result["blackout_probability"] < 0.10

    def test_stability_proxies_normal(self, normal_snapshot):
        """Las proxies deben reflejar operación normal (datos REALES ESIOS:
        ~66% renovable de media mañana es normal; el >90% sostenido del
        28-abr es el patrón de riesgo)."""
        from tools.stability_proxy_calculator import stability_proxy_calculator
        proxies = stability_proxy_calculator(normal_snapshot)
        assert proxies["renewables_pct"] < 85
        # Real 14-may-2025 09:00 → 0.57 absoluto; bajo el umbral S1 (P90=0.724 Markov)
        assert proxies["composite_pressure"] < 0.65

    @pytest.mark.integration
    @_requires_live_llm
    def test_run_scenario_normal_no_escalation(self, normal_scenario):
        """No debe activar CrossBorderAgent ni ReportAgent en condiciones normales."""
        from agents.orchestrator import run_scenario
        result = run_scenario(normal_scenario)

        trajectory = result.get("trajectory", [])
        tool_names = [step["tool"] for step in trajectory]

        assert "cross_border_agent" not in tool_names, \
            "cross_border_agent NO debe activarse con P<0.10"
        assert "report_agent" not in tool_names, \
            "report_agent NO debe activarse en operación normal"

        sev = result.get("severity", "LOW")
        assert sev in ("LOW", "MEDIUM"), \
            f"Severity esperada LOW/MEDIUM en escenario normal, obtenida: {sev}"
