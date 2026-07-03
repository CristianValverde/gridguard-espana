"""_embed_text debe acotar la llamada HTTP a la API de embeddings de Gemini.

Regresión live 2026-07-03: sin timeout, una llamada lenta (p.ej. bajo
throttling de cuota) bloquea el event loop del kb server más allá de los
60s del cliente MCP y encola al resto de requests.
"""
from __future__ import annotations

from types import SimpleNamespace


def test_embed_text_usa_timeout_http_acotado(monkeypatch):
    captured: dict = {}

    class _FakeModels:
        def embed_content(self, **kwargs):
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.0, 1.0])])

    class _FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.models = _FakeModels()

    import google.genai as genai_mod

    monkeypatch.setattr(genai_mod, "Client", _FakeClient)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    from mcp_servers import knowledge_base_server as kb

    result = kb._embed_text(["hola"])
    assert result == [[0.0, 1.0]]

    http_options = captured.get("http_options")
    assert http_options is not None, "genai.Client sin http_options: llamada sin timeout"
    timeout_ms = getattr(http_options, "timeout", None)
    if timeout_ms is None and isinstance(http_options, dict):
        timeout_ms = http_options.get("timeout")
    assert timeout_ms is not None, "http_options sin timeout"
    assert 0 < timeout_ms <= 60_000, f"timeout fuera de rango: {timeout_ms} ms"
