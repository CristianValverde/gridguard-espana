"""Tests for state transition hooks (Task 2)."""
from guardrails import hooks


def test_recovery_to_normal_triggers_post_mortem():
    """S2+ → S0 should trigger post_mortem action."""
    result = hooks.on_state_transition(prev_state=3, curr_state=0)
    assert result["action"] == "post_mortem"
    assert result["trigger"] == "S3->S0"


def test_onset_from_normal_triggers_regulatory_check():
    """S0 → S2+ should trigger regulatory_check action."""
    result = hooks.on_state_transition(prev_state=0, curr_state=2)
    assert result["action"] == "regulatory_check"
    assert result["trigger"] == "S0->S2"


def test_minor_change_is_noop():
    """Minor state changes (S0→S1, S1→S2) should be noop."""
    result = hooks.on_state_transition(prev_state=0, curr_state=1)
    assert result["action"] == "none"
    assert result["trigger"] is None


def test_s1_to_s0_is_noop():
    """S1→S0 is a noop (not a notable recovery)."""
    result = hooks.on_state_transition(prev_state=1, curr_state=0)
    assert result["action"] == "none"


def test_forecast_risk_horizon_logs_window():
    """Forecast risk horizon should log window and return advance_warning."""
    result = hooks.on_forecast_risk_horizon("14:00-14:30 UTC", "HIGH")
    assert result["logged"] is True
    assert result["advance_warning"] == "14:00-14:30 UTC"
