"""chromadb debe importarse EAGER en el arranque del kb server.

Regresión live 2026-07-04 (py-spy): el import perezoso de chromadb (que
arrastra numpy) dentro del primer tool call, con el event loop de anyio ya
corriendo, se colgaba minutos en la carga de la extensión C de numpy y el
cliente MCP agotaba su timeout. Mismo patrón que torch en ml_models_server:
el import pesado se paga una vez en el arranque del proceso (~1.5s).
"""
from __future__ import annotations

import inspect

from mcp_servers import knowledge_base_server as kb


def test_chromadb_importado_eager_en_el_modulo():
    assert kb.chromadb is not None, "chromadb debe importarse al cargar el módulo"


def test_get_chroma_no_hace_import_lazy():
    src = inspect.getsource(kb._get_chroma)
    assert "import chromadb" not in src, "el import debe vivir a nivel de módulo"
