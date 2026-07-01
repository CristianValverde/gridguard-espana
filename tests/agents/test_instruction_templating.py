"""Verifica que las instrucciones de los agentes NO sufren el template-injection de ADK 2.3.0.

ADK 2.3.0 sustituye `{variable}` en una `instruction` de tipo str con valores de
session-state, lanzando KeyError con las llaves literales de nuestros prompts
(ej. `{risk_score}`, ejemplos JSON). Envolver la instrucción en un InstructionProvider
(callable) hace que `canonical_instruction` devuelva bypass_state_injection=True y ADK
NO aplique la sustitución. Ver agents/instruction_utils.py.
"""
import asyncio

from agents.instruction_utils import as_static_instruction


def test_as_static_instruction_returns_provider_that_preserves_braces():
    provider = as_static_instruction("texto con {llaves_literales} y {P:.1%}")
    assert callable(provider)
    # el provider ignora el contexto y devuelve el texto literal, con las llaves intactas
    assert provider(None) == "texto con {llaves_literales} y {P:.1%}"


def test_orchestrator_instruction_bypasses_template_injection():
    from agents.orchestrator import orchestrator

    text, bypass = asyncio.run(orchestrator.canonical_instruction(None))
    # InstructionProvider → ADK marca bypass y NO templatea
    assert bypass is True
    # las llaves literales del prompt se preservan (no habría llegado aquí si ADK las sustituyera)
    assert "{" in text


def test_all_agent_instructions_are_providers_not_raw_strings():
    """Cada Agent debe llevar la instrucción como callable, no como str (que se templataría)."""
    from agents.orchestrator import orchestrator
    from agents.report_agent import report_agent
    from agents.cross_border import cross_border_agent
    from agents.regulatory_compliance import _inner_agent as reg_agent
    from agents.anomaly_detector import _inner_agent as anom_agent
    from agents.renewables_forecast import _inner_agent as ren_agent

    for agent in (orchestrator, report_agent, cross_border_agent, reg_agent, anom_agent, ren_agent):
        assert not isinstance(agent.instruction, str), f"{agent.name}: instrucción es str (se templataría)"
        assert callable(agent.instruction), f"{agent.name}: instrucción no es un InstructionProvider"
