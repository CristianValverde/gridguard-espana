"""Tests for tool whitelist authorization (Task 3)."""
from guardrails.hooks import _ALLOWED_TOOLS


def test_orchestrator_can_use_grid_state_and_compliance():
    """Orchestrator should have access to grid state tools and compliance skill."""
    o = _ALLOWED_TOOLS["gridguard_orchestrator"]
    assert "get_grid_state" in o
    assert "RegulatoryComplianceSkill" in o


def test_anomaly_can_use_markov_and_weather():
    """Anomaly detector should have access to grid state and weather tools."""
    a = _ALLOWED_TOOLS["anomaly_detector_skill"]
    assert {"get_grid_state", "get_wind_forecast", "get_solar_radiation"} <= a


def test_renewables_can_use_24h_forecast():
    """Renewables forecast skill should have access to 24h weather forecast."""
    assert "get_weather_forecast_24h" in _ALLOWED_TOOLS["renewables_forecast_skill"]


def test_report_can_use_trajectory_and_transitions():
    """Report agent should have access to markov trajectory and state transitions."""
    r = _ALLOWED_TOOLS["report_agent"]
    assert {"get_markov_trajectory", "get_state_transitions"} <= r


def test_cross_border_can_use_market_prices():
    """Cross-border agent should have access to EU market prices."""
    assert "get_eu_market_prices" in _ALLOWED_TOOLS["cross_border_agent"]


def test_regulatory_compliance_whitelist_exists():
    """Regulatory compliance skill should have its own whitelist with search_regulations and get_grid_state."""
    rc = _ALLOWED_TOOLS["regulatory_compliance_skill"]
    assert {"search_regulations", "get_grid_state"} <= rc
