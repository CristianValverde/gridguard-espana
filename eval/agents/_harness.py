"""Harness compartido para evals live (integration): ejecuta un Agent ADK
con un mensaje y devuelve output final + trayectoria de tools.

Generaliza el patrón de agents.orchestrator.run_scenario para poder evaluar
cualquier agente (incluidos los inner agents de las Skills) individualmente.
"""
from __future__ import annotations

import asyncio
from typing import Any

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from agents.orchestrator import _extract_result, _extract_trajectory


def run_agent_live(agent, message_text: str) -> dict[str, Any]:
    """Ejecuta `agent` (google.adk Agent) con `message_text` y devuelve
    {"output": dict, "trajectory": list[{"tool": str, ...}]}."""
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="gridguard_eval", session_service=session_service)
    session = asyncio.run(
        session_service.create_session(app_name="gridguard_eval", user_id="eval-runner")
    )
    message = genai_types.Content(role="user", parts=[genai_types.Part(text=message_text)])
    events = list(runner.run(session_id=session.id, user_id="eval-runner", new_message=message))
    return {"output": _extract_result(events), "trajectory": _extract_trajectory(events)}
