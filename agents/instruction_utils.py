"""
agents/instruction_utils.py — Helper para instrucciones estáticas en ADK 2.3.0.

ADK 2.3.0 procesa una `instruction` de tipo `str` como plantilla e intenta sustituir
`{variable}` con valores de session-state (ver LlmAgent.canonical_instruction ->
inject_session_state). Nuestras instrucciones contienen llaves literales (ejemplos JSON,
formatos como `{risk_score}`) que NO son variables de estado, y eso provoca
`KeyError: 'Context variable not found: ...'` al ejecutar el agente.

canonical_instruction devuelve `bypass_state_injection=True` cuando `instruction` es un
`InstructionProvider` (un callable), en cuyo caso ADK NO aplica el template injection.
Este helper envuelve el texto literal en ese callable, preservando las llaves tal cual.
"""

from __future__ import annotations

from typing import Callable


def as_static_instruction(text: str) -> Callable:
    """Envuelve una instrucción literal como InstructionProvider (bypass de templating ADK 2.3.0).

    Uso en un agente:
        from agents.instruction_utils import as_static_instruction
        Agent(..., instruction=as_static_instruction(_INSTRUCTION), ...)
    """

    def _provider(readonly_context) -> str:  # noqa: ANN001 — firma InstructionProvider de ADK
        return text

    return _provider
