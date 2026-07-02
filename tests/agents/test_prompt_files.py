"""Prompts versionables: un fichero por agente, cargado en import, patron intacto."""
from __future__ import annotations

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_AGENTS = ["orchestrator", "anomaly_detector", "renewables_forecast",
           "regulatory_compliance", "report_agent", "cross_border"]


@pytest.mark.parametrize("name", _AGENTS)
def test_prompt_file_exists_and_non_empty(name):
    p = _ROOT / "prompts" / f"{name}.md"
    assert p.exists(), f"falta {p}"
    assert p.read_text(encoding="utf-8").strip(), f"prompt vacio: {p}"


@pytest.mark.parametrize("name", _AGENTS)
def test_agent_module_loads_prompt_from_file(name):
    src = (_ROOT / "agents" / f"{name}.py").read_text(encoding="utf-8")
    assert f'load_prompt("{name}")' in src, f"{name} no carga su prompt desde prompts/"
    assert '_INSTRUCTION = """' not in src, f"{name} conserva el literal inline"


def test_load_prompt_roundtrip():
    from agents.instruction_utils import load_prompt
    text = load_prompt("orchestrator")
    assert "PROTOCOLO DE MONITORIZACION" in text
    assert text == (_ROOT / "prompts" / "orchestrator.md").read_text(encoding="utf-8").strip()
