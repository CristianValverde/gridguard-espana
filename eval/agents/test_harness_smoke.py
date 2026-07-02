"""El harness debe importar y exponer la firma esperada sin LLM (smoke offline)."""
import inspect


def test_harness_signature():
    from eval.agents import _harness
    sig = inspect.signature(_harness.run_agent_live)
    assert list(sig.parameters) == ["agent", "message_text"]
