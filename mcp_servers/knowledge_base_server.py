"""
mcp_servers/knowledge_base_server.py — MCP: knowledge-base

RAG sobre documentos técnicos: normas REE/ENTSO-E, post-mortems de incidentes,
procedimientos operacionales.

Vector store: ChromaDB local.
Embeddings: gemini-embedding-2 via Google Generative AI SDK.

Herramientas expuestas:
  search_regulations(query, top_k)         -> list[RegulatoryChunk] (JSON)
  find_similar_incident(description, top_k) -> list[IncidentRecord] (JSON)
  get_mitigation_procedure(situation)       -> Procedure (JSON)

Inicialización: la KB debe haberse construido con knowledge/build_kb.py.
Ruta ChromaDB: variable de entorno CHROMA_DB_PATH (default: ./knowledge/chroma_db).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from mcp.server.fastmcp import FastMCP

from config.model_loader import get_model
from schemas.models import IncidentRecord, Procedure, RegulatoryChunk

mcp = FastMCP("gridguard-knowledge-base")

_DEFAULT_CHROMA = "./knowledge/chroma_db"

# Nombres de colecciones ChromaDB
COLLECTION_REGULATIONS = "regulations"
COLLECTION_INCIDENTS = "incidents"
COLLECTION_PROCEDURES = "procedures"

_chroma_client: Any = None
_embed_model: str = ""


def _chroma_path() -> str:
    return os.environ.get("CHROMA_DB_PATH", _DEFAULT_CHROMA)


def _get_chroma():
    """Crea (o reutiliza) el cliente ChromaDB. Lazy init."""
    global _chroma_client
    if _chroma_client is not None:
        return _chroma_client
    try:
        import chromadb
        client = chromadb.PersistentClient(path=_chroma_path())
        _chroma_client = client
        return client
    except ImportError as exc:
        raise ImportError("chromadb no instalado. pip install chromadb") from exc


def _embed_text(texts: list[str]) -> list[list[float]]:
    """Genera embeddings usando gemini-embedding-2 (google.genai SDK)."""
    global _embed_model
    if not _embed_model:
        _embed_model = get_model("embedding")

    try:
        from google import genai
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        result = client.models.embed_content(
            model=_embed_model,
            contents=texts,
        )
        return [e.values for e in result.embeddings]
    except Exception as exc:
        raise RuntimeError(f"Error generando embeddings con {_embed_model}: {exc}") from exc


def _query_collection(
    collection_name: str,
    query: str,
    top_k: int,
) -> list[dict]:
    """Consulta una colección ChromaDB y devuelve los documentos más similares."""
    try:
        client = _get_chroma()
        collection = client.get_collection(collection_name)
    except Exception as exc:
        return [{"error": f"Colección '{collection_name}' no encontrada. Ejecutar knowledge/build_kb.py. Detalle: {exc}"}]

    try:
        embeddings = _embed_text([query])
        # Si embed_text devuelve lista de listas o lista de floats, normalizar
        if embeddings and isinstance(embeddings[0], float):
            query_embedding = [embeddings]
        else:
            query_embedding = [embeddings[0]]

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        return results
    except Exception as exc:
        return [{"error": f"Error consultando ChromaDB: {exc}"}]


# ---------------------------------------------------------------------------
# Herramientas MCP
# ---------------------------------------------------------------------------

@mcp.tool()
def search_regulations(query: str, top_k: int = 5) -> list[dict]:
    """Busca fragmentos relevantes en la normativa técnica (ENTSO-E, POs REE, EN 50160).

    Args:
        query:  Pregunta o descripción de la situación operacional
        top_k:  Número de fragmentos a devolver (default 5, máx 10)
    """
    top_k = max(1, min(top_k, 10))
    raw = _query_collection(COLLECTION_REGULATIONS, query, top_k)

    if isinstance(raw, list) and raw and "error" in raw[0]:
        return raw

    chunks: list[dict] = []
    documents = raw.get("documents", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        try:
            chunk = RegulatoryChunk(
                chunk_id=meta.get("chunk_id", "unknown"),
                document=meta.get("document", "unknown"),
                section=meta.get("section"),
                content=doc,
                relevance_score=round(1.0 - float(dist), 4),
            )
            chunks.append(chunk.model_dump(mode="json"))
        except Exception:
            continue
    return chunks


@mcp.tool()
def find_similar_incident(description: str, top_k: int = 3) -> list[dict]:
    """Busca post-mortems históricos similares a la situación descrita.

    Post-mortems disponibles:
    - Apagón España 28 de abril de 2025 (REE)
    - Separación ibérica 2006
    - Incidentes CEER 2018-2024

    Args:
        description: Descripción de la situación anómala actual
        top_k:       Número de incidentes similares a devolver (default 3)
    """
    top_k = max(1, min(top_k, 5))
    raw = _query_collection(COLLECTION_INCIDENTS, description, top_k)

    if isinstance(raw, list) and raw and "error" in raw[0]:
        return raw

    incidents: list[dict] = []
    documents = raw.get("documents", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        try:
            incident = IncidentRecord(
                incident_id=meta.get("incident_id", "unknown"),
                date=datetime.fromisoformat(
                    meta.get("date", datetime.now(timezone.utc).isoformat())
                ),
                title=meta.get("title", "Sin título"),
                description=doc,
                cause=meta.get("cause", "Causa no documentada"),
                outcome=meta.get("outcome", "Outcome no documentado"),
                similarity_score=round(1.0 - float(dist), 4),
            )
            incidents.append(incident.model_dump(mode="json"))
        except Exception:
            continue
    return incidents


@mcp.tool()
def get_mitigation_procedure(situation: str) -> dict:
    """Recupera el procedimiento operacional de REE más relevante para la situación dada.

    Procedimientos indexados: PO 1.1, PO 1.2, PO 7.4, protocolo reducción carga,
    protocolo separación de red, procedimiento importación emergencia.

    Args:
        situation: Descripción breve de la situación (ej. "alta penetración renovable con demanda baja")
    """
    raw = _query_collection(COLLECTION_PROCEDURES, situation, top_k=1)

    if isinstance(raw, list) and raw and "error" in raw[0]:
        return raw[0]

    documents = raw.get("documents", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]

    if not documents:
        return {"error": "No se encontró procedimiento relevante. Verificar que la KB está construida."}

    doc = documents[0]
    meta = metadatas[0]

    try:
        # Los pasos del procedimiento se almacenan como JSON en el documento
        import json as _json
        steps_data = _json.loads(doc) if doc.startswith("[") else [doc]
    except Exception:
        steps_data = [doc]

    try:
        return Procedure(
            procedure_id=meta.get("procedure_id", "unknown"),
            title=meta.get("title", "Procedimiento operacional"),
            situation=meta.get("situation", situation),
            steps=steps_data if isinstance(steps_data, list) else [str(steps_data)],
            authority=meta.get("authority", "REE"),
        ).model_dump(mode="json")
    except Exception as exc:
        return {"error": f"Error construyendo Procedure: {exc}"}


if __name__ == "__main__":
    mcp.run()
