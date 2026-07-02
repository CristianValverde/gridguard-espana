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

from pathlib import Path
from typing import Callable

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(agent_name: str) -> str:
    """Lee prompts/<agent_name>.md (UTF-8) y devuelve el texto listo para
    as_static_instruction. Falla alto y pronto si falta o esta vacio."""
    path = _PROMPTS_DIR / f"{agent_name}.md"
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt vacío: {path}")
    return text


def as_static_instruction(text: str) -> Callable:
    """Envuelve una instrucción literal como InstructionProvider (bypass de templating ADK 2.3.0).

    Uso en un agente:
        from agents.instruction_utils import as_static_instruction
        Agent(..., instruction=as_static_instruction(_INSTRUCTION), ...)
    """

    def _provider(readonly_context) -> str:  # noqa: ANN001 — firma InstructionProvider de ADK
        return text

    return _provider
