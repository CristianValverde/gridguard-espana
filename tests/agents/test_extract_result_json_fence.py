"""_extract_result debe reconocer JSON envuelto en fences de markdown.

Regresion descubierta en el smoke test del capstone (2026-07-04, tras
actualizar a gemini-3.5-flash): el orchestrator devuelve su JSON final
envuelto en ```json ... ``` en vez de texto plano empezando por "{",
y _extract_result caia al fallback {"raw": texto_completo} en vez de
parsear el dict real.
"""
from __future__ import annotations

from types import SimpleNamespace

from agents.orchestrator import _extract_result


def _fake_event(text: str):
    part = SimpleNamespace(text=text)
    content = SimpleNamespace(parts=[part])
    return SimpleNamespace(
        content=content,
        is_final_response=lambda: True,
    )


def test_extrae_json_envuelto_en_fence_con_lenguaje():
    events = [_fake_event('```json\n{"severity": "CRITICAL", "risk_score": 90}\n```')]
    result = _extract_result(events)
    assert result == {"severity": "CRITICAL", "risk_score": 90}


def test_extrae_json_envuelto_en_fence_sin_lenguaje():
    events = [_fake_event('```\n{"severity": "LOW"}\n```')]
    result = _extract_result(events)
    assert result == {"severity": "LOW"}


def test_sigue_extrayendo_json_plano_sin_fence():
    events = [_fake_event('{"severity": "MEDIUM"}')]
    result = _extract_result(events)
    assert result == {"severity": "MEDIUM"}


def test_texto_no_json_sigue_cayendo_en_raw():
    events = [_fake_event("no puedo procesar esta solicitud")]
    result = _extract_result(events)
    assert result == {"raw": "no puedo procesar esta solicitud"}
