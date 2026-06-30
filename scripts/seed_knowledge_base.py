"""
scripts/seed_knowledge_base.py — Puebla ChromaDB con regulaciones REE e incidentes.

Usa Gemini embeddings para vectorizar. Solo requiere GEMINI_API_KEY.
Sin ESIOS ni ENTSO-E.

Uso:
  $env:GEMINI_API_KEY="tu_clave"
  python scripts/seed_knowledge_base.py [--reset]

  --reset  : Borra las colecciones existentes antes de insertar (re-seed completo)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

SEED_DIR = ROOT / "data" / "kb_seed"

# Mapa: carpeta → colección ChromaDB
COLLECTION_MAP = {
    "regulations": "regulations",
    "incidents": "incidents",
    "procedures": "procedures",
}

CHUNK_SIZE = 600       # chars por chunk
CHUNK_OVERLAP = 100    # chars de overlap entre chunks
RATE_LIMIT_SLEEP = 1.0 # segundos entre llamadas a embedding API


def _api_key() -> str:
    k = os.environ.get("GEMINI_API_KEY")
    if not k:
        sys.exit("ERROR: GEMINI_API_KEY no definida.")
    return k


def _get_chroma_client():
    try:
        import chromadb
        # Misma ruta que usa knowledge_base_server.py (CHROMA_DB_PATH o ./knowledge/chroma_db)
        chroma_dir = Path(os.environ.get("CHROMA_DB_PATH", str(ROOT / "knowledge" / "chroma_db")))
        chroma_dir.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(chroma_dir))
    except ImportError:
        sys.exit("ERROR: pip install chromadb")


def _get_embedding_fn(api_key: str):
    """Crea función de embedding usando google.genai SDK."""
    try:
        from google import genai
    except ImportError:
        sys.exit("ERROR: pip install google-genai")

    client = genai.Client(api_key=api_key)

    from config.model_loader import get_model
    embedding_model = get_model("embedding")

    def embed(texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            result = client.models.embed_content(
                model=embedding_model,
                contents=[text],
            )
            embeddings.append(result.embeddings[0].values)
            time.sleep(RATE_LIMIT_SLEEP)
        return embeddings

    return embed, embedding_model


def chunk_text(text: str, source_file: str) -> list[dict]:
    """Divide el texto en chunks con overlap."""
    chunks = []
    start = 0
    chunk_idx = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():
            chunks.append({
                "text": chunk.strip(),
                "id": f"{Path(source_file).stem}_chunk{chunk_idx:03d}",
                "source": Path(source_file).name,
                "chunk_index": chunk_idx,
            })
        start += CHUNK_SIZE - CHUNK_OVERLAP
        chunk_idx += 1
    return chunks


def seed_collection(client, collection_name: str, folder: Path,
                    embed_fn, reset: bool = False) -> int:
    """Puebla una colección de ChromaDB con los .txt de la carpeta."""
    print(f"\n{'─'*50}")
    print(f"Colección: {collection_name}  →  {folder}")
    print(f"{'─'*50}")

    if reset:
        try:
            client.delete_collection(collection_name)
            print(f"  Colección {collection_name} borrada (--reset).")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    txt_files = sorted(folder.glob("*.txt"))
    if not txt_files:
        print(f"  No hay archivos .txt en {folder}")
        return 0

    total_chunks = 0
    for txt_file in txt_files:
        print(f"  Procesando: {txt_file.name}")
        text = txt_file.read_text(encoding="utf-8")
        chunks = chunk_text(text, str(txt_file))

        # Verificar si ya existen (skip si no hay --reset)
        existing_ids = set(collection.get(ids=[c["id"] for c in chunks])["ids"])
        new_chunks = [c for c in chunks if c["id"] not in existing_ids]

        if not new_chunks:
            print(f"    ↳ Ya indexado ({len(chunks)} chunks). Saltando.")
            continue

        print(f"    ↳ {len(new_chunks)} chunks nuevos a indexar…")
        texts = [c["text"] for c in new_chunks]
        embeddings = embed_fn(texts)

        collection.add(
            ids=[c["id"] for c in new_chunks],
            documents=[c["text"] for c in new_chunks],
            embeddings=embeddings,
            metadatas=[{"source": c["source"], "chunk_index": c["chunk_index"]}
                       for c in new_chunks],
        )
        total_chunks += len(new_chunks)
        print(f"    ↳ Indexados {len(new_chunks)} chunks con embeddings.")

    total_in_collection = collection.count()
    print(f"  Total en colección '{collection_name}': {total_in_collection} chunks.")
    return total_chunks


def test_retrieval(client, embed_fn) -> None:
    """Prueba que el RAG devuelve resultados relevantes para queries clave."""
    print(f"\n{'─'*50}")
    print("TEST DE RECUPERACIÓN RAG")
    print(f"{'─'*50}")

    test_queries = [
        ("regulations", "límites de frecuencia sistema eléctrico", "PO_7_1"),
        ("incidents", "apagón alta penetración renovable abril 2025", "apagon_abril"),
        ("procedures", "importación emergencia España Francia", "importacion"),
    ]

    for collection_name, query, expected_source_kw in test_queries:
        collection = client.get_or_create_collection(collection_name)
        if collection.count() == 0:
            print(f"  [SKIP] {collection_name} vacía")
            continue

        query_emb = embed_fn([query])[0]
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=2,
            include=["documents", "metadatas", "distances"],
        )

        doc = results["documents"][0][0][:120] if results["documents"][0] else "(vacío)"
        dist = results["distances"][0][0] if results["distances"][0] else 1.0
        meta = results["metadatas"][0][0] if results["metadatas"][0] else {}

        ok = expected_source_kw.lower() in meta.get("source", "").lower()
        status = "✓" if ok else "?"
        print(f"  {status} [{collection_name}] '{query[:40]}…'")
        print(f"    Fuente: {meta.get('source', 'N/A')}  |  Distancia: {dist:.3f}")
        print(f"    Fragmento: {doc}…")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true",
                        help="Borra las colecciones existentes antes de re-indexar")
    args = parser.parse_args()

    api_key = _api_key()
    print("Iniciando seed de base de conocimiento GridGuard…")
    print(f"Directorio seed: {SEED_DIR}")

    embed_fn, model_name = _get_embedding_fn(api_key)
    print(f"Modelo de embedding: {model_name}")

    client = _get_chroma_client()

    total = 0
    for folder_name, collection_name in COLLECTION_MAP.items():
        folder = SEED_DIR / folder_name
        if not folder.exists():
            print(f"  Carpeta no encontrada: {folder} — saltando.")
            continue
        n = seed_collection(client, collection_name, folder, embed_fn, reset=args.reset)
        total += n

    print(f"\n{'='*50}")
    print(f"SEED COMPLETADO: {total} chunks nuevos indexados.")
    print(f"{'='*50}")

    if total > 0 or args.reset:
        test_retrieval(client, embed_fn)
    elif total == 0:
        print("\nTodos los documentos ya estaban indexados. Usa --reset para re-indexar.")
        test_retrieval(client, embed_fn)

    print("\nListo. El servidor MCP knowledge-base puede arrancar con:")
    print("  python -m mcp_servers.knowledge_base_server")


if __name__ == "__main__":
    main()
