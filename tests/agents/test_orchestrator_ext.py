"""
tests/agents/test_orchestrator_ext.py

Plan 4 Task 8 — OrchestratorAgent:
  - Migrado a ADK 2.3.0 (sin temperature ni max_turns como kwargs directos)
  - Estado de Markov (get_grid_state) incorporado en PASO 1
  - Nuevos umbrales combinados (Markov + P(blackout))
  - RegulatoryComplianceSkill en tramo NARANJA
"""

import importlib
import sys


def _fresh_import():
    """Reimporta el modulo para obtener el estado mas reciente.

    ADK 2.3.0 asigna parent_agent en los sub-agents cuando se agregan a un Agent.
    Para evitar el ValidationError 'already has a parent agent' en importaciones
    repetidas (p.ej. entre tests del mismo proceso), se eliminan tambien los modulos
    de los sub-agents y skills para que se recreen sus instancias desde cero.
    """
    _agent_modules = [
        "agents.orchestrator",
        "agents.report_agent",
        "agents.cross_border",
        "agents.anomaly_detector",
        "agents.renewables_forecast",
        "agents.regulatory_compliance",
    ]
    for mod in _agent_modules:
        sys.modules.pop(mod, None)
    return importlib.import_module("agents.orchestrator")


def test_instruction_has_markov_and_new_thresholds():
    """La instruccion del orchestrator debe mencionar:
    - get_grid_state (adquisicion de estado Markov en PASO 1)
    - RegulatoryComplianceSkill (invocada en tramo NARANJA)
    - assessment (CrossBorderAgent en modo assessment)
    - S2 (umbral de override por estado Markov)
    """
    orch = _fresh_import()
    text = orch._INSTRUCTION

    assert "get_grid_state" in text, (
        "La instruccion debe incluir 'get_grid_state' en PASO 1"
    )
    assert "RegulatoryComplianceSkill" in text, (
        "La instruccion debe mencionar 'RegulatoryComplianceSkill' en tramo NARANJA"
    )
    assert "assessment" in text, (
        "La instruccion debe mencionar CrossBorderAgent en modo 'assessment'"
    )
    assert "S2" in text, (
        "La instruccion debe mencionar el umbral Markov 'S2' para el override"
    )


def test_module_imports():
    """El modulo debe importarse correctamente bajo ADK 2.3.0
    (sin ValidationError por temperature/max_turns, sin AttributeError en register_hooks).
    """
    orch = _fresh_import()
    assert orch is not None
    # El agente orquestador debe existir y tener nombre
    assert hasattr(orch, "orchestrator")
    assert orch.orchestrator.name == "gridguard_orchestrator"
