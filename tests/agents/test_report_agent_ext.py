# tests/agents/test_report_agent_ext.py


def test_instruction_supports_post_mortem():
    import agents.report_agent as ra
    assert "post_mortem" in ra._INSTRUCTION
    assert "get_state_transitions" in ra._INSTRUCTION
    assert "get_markov_trajectory" in ra._INSTRUCTION


def test_module_imports():
    import agents.report_agent as ra
    assert ra.report_agent is not None
