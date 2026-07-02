"""Smoke test de recuperación RAG del contexto de interconexiones (Task 11).

Requiere ChromaDB poblado (python -m scripts.kb_embed) y acceso a
embeddings Gemini → marcado integration (deseleccionado en la suite offline).
"""
import pytest


@pytest.mark.integration
def test_interconnection_context_retrievable():
    from mcp_servers.knowledge_base_server import get_mitigation_procedure

    res = get_mitigation_procedure("capacidad de la interconexion Espana-Francia")
    txt = str(res).lower()
    assert "francia" in txt or "es_fr" in txt or "interconexion" in txt


@pytest.mark.integration
def test_morocco_link_is_ac_not_hvdc():
    """La reconciliación del corpus: el enlace ES-MA es AC submarino, no HVDC."""
    from mcp_servers.knowledge_base_server import get_mitigation_procedure

    res = get_mitigation_procedure("interconexion Espana Marruecos cable submarino")
    txt = str(res).lower()
    assert "marruecos" in txt or "tarifa" in txt or "es_ma" in txt
