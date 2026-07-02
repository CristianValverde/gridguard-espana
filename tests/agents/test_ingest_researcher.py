from pathlib import Path

import pytest
import yaml


def test_topics_yaml_parses_with_required_fields():
    topics = yaml.safe_load(Path("data/kb_seed/topics.yaml").read_text(encoding="utf-8"))
    assert isinstance(topics, list) and topics
    for t in topics:
        assert {"id", "collection", "title", "status"} <= set(t.keys())
        assert t["collection"] in ("regulations", "procedures", "incidents")
        assert t["status"] in ("pending", "generated")


def test_researcher_builds_with_model_from_registry():
    from agents.ingest.researcher import build_researcher
    from config.model_loader import get_model

    agent = build_researcher()
    assert agent.name == "kb_researcher"
    assert agent.model == get_model("smart_agent")  # nunca hardcodeado


def test_instruction_contract():
    """Las reglas críticas del diseño deben estar en la instrucción."""
    from agents.ingest.researcher import _INSTRUCTION

    assert "[SIN VERIFICAR]" in _INSTRUCTION
    assert "FUENTES" in _INSTRUCTION
    assert "google_search" in _INSTRUCTION or "búsqueda" in _INSTRUCTION.lower()
    assert "MAYÚSCULAS" in _INSTRUCTION


def test_extract_text_takes_last_text_event():
    from agents.ingest.run_researcher import _extract_text

    class _Part:
        def __init__(self, text):
            self.text = text

    class _Content:
        def __init__(self, text):
            self.parts = [_Part(text)]

    class _Event:
        def __init__(self, text):
            self.content = _Content(text) if text else None

    events = [_Event("intermedio"), _Event(None), _Event("DOCUMENTO FINAL\n\nFUENTES\n- x")]
    assert _extract_text(events).startswith("DOCUMENTO FINAL")
