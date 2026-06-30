"""
tests/agents/test_renewables_forecast_ext.py
TDD tests for Task 5: ADK 2.3.0 migration + 24h horizon + Markov context.
"""


def test_instruction_mentions_24h_and_markov():
    """Instrucción debe mencionar get_weather_forecast_24h y get_grid_state."""
    import agents.renewables_forecast as rf

    assert "get_weather_forecast_24h" in rf._INSTRUCTION, (
        "_INSTRUCTION debe referenciar get_weather_forecast_24h (horizonte 24h)"
    )
    assert "get_grid_state" in rf._INSTRUCTION, (
        "_INSTRUCTION debe referenciar get_grid_state (contexto Markov)"
    )


def test_module_imports():
    """El módulo debe importar correctamente bajo ADK 2.3.0."""
    import agents.renewables_forecast as rf

    assert rf.RenewablesForecastSkill is not None, (
        "RenewablesForecastSkill debe existir y no ser None"
    )
