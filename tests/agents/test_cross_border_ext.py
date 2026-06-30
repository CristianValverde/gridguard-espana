"""
tests/agents/test_cross_border_ext.py

Plan 4 · Task 7 — CrossBorderAgent: ADK 2.3.0 migration + mode parameter + EU market prices.
"""


def test_instruction_supports_mode_and_prices():
    import agents.cross_border as cb

    assert "assessment" in cb._INSTRUCTION, "'assessment' debe estar en _INSTRUCTION"
    assert "emergency" in cb._INSTRUCTION, "'emergency' debe estar en _INSTRUCTION"
    assert "get_eu_market_prices" in cb._INSTRUCTION, "'get_eu_market_prices' debe estar en _INSTRUCTION"


def test_module_imports():
    import agents.cross_border as cb

    assert cb.cross_border_agent is not None, "cross_border_agent debe existir y no ser None"
