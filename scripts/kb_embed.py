"""scripts/kb_embed.py — Indexado idempotente del corpus RAG en ChromaDB.

Reemplaza a scripts/seed_knowledge_base.py. Diferencias:
  - IDs de chunk direccionados por contenido (stem + índice + sha8) → upsert
    idempotente: re-ejecutar no duplica; contenido cambiado re-indexa; chunks
    obsoletos y fuentes retiradas se podan.
  - Solo indexa documentos con status approved/indexed en el manifiesto.
  - Reintentos con backoff en la llamada de embeddings.
Python puro salvo la llamada de embeddings (get_model("embedding")).

Uso: python -m scripts.kb_embed [--seed-dir data/kb_seed] [--manifest ...]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from scripts import kb_manifest as M

COLLECTION_MAP = {
    "regulations": "regulations",
    "incidents": "incidents",
    "procedures": "procedures",
}
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
RATE_LIMIT_SLEEP = 1.0


def chunk_text(text: str, source_name: str) -> list[dict]:
    """Chunks 600/100 con ID direccionado por contenido (estable e idempotente)."""
    stem = Path(source_name).stem
    chunks: list[dict] = []
    start = 0
    idx = 0
    while start < len(text):
        piece = text[start:start + CHUNK_SIZE].strip()
        if piece:
            sha8 = M.compute_sha256(piece)[:8]
            chunks.append({
                "id": f"{stem}_c{idx:03d}_{sha8}",
                "text": piece,
                "source": Path(source_name).name,
                "chunk_index": idx,
            })
        start += CHUNK_SIZE - CHUNK_OVERLAP
        idx += 1
    return chunks


def _embed_with_retry(embed_fn, texts: list[str], attempts: int = 3,
                      base_sleep: float = 2.0) -> list[list[float]]:
    for i in range(attempts):
        try:
            return embed_fn(texts)
        except Exception as exc:
            if i == attempts - 1:
                raise
            wait = base_sleep * (2 ** i)
            print(f"  embeddings fallaron ({exc}); reintento en {wait:.0f}s…")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def upsert_document(collection, txt_path: Path, embed_fn) -> tuple[int, int]:
    """Sincroniza un .txt con la colección. Devuelve (añadidos, podados)."""
    text = Path(txt_path).read_text(encoding="utf-8")
    chunks = chunk_text(text, Path(txt_path).name)
    existing = collection.get(where={"source": Path(txt_path).name})
    existing_ids = set(existing["ids"])
    new_ids = {c["id"] for c in chunks}

    stale = sorted(existing_ids - new_ids)
    if stale:
        collection.delete(ids=stale)

    to_add = [c for c in chunks if c["id"] not in existing_ids]
    if to_add:
        embeddings = _embed_with_retry(embed_fn, [c["text"] for c in to_add])
        collection.add(
            ids=[c["id"] for c in to_add],
            documents=[c["text"] for c in to_add],
            embeddings=embeddings,
            metadatas=[{"source": c["source"], "chunk_index": c["chunk_index"]}
                       for c in to_add],
        )
    return len(to_add), len(stale)


def prune_orphan_sources(collection, approved_names: set[str]) -> int:
    """Poda chunks cuya fuente ya no está aprobada/presente."""
    data = collection.get(include=["metadatas"])
    orphans = [cid for cid, meta in zip(data["ids"], data["metadatas"])
               if meta.get("source") not in approved_names]
    if orphans:
        collection.delete(ids=orphans)
    return len(orphans)


def run(seed_dir: Path, manifest_path: Path, client, embed_fn) -> int:
    seed_dir = Path(seed_dir)
    manifest = M.load_manifest(manifest_path)
    docs = manifest.get("documents", {})
    total_added = total_pruned = 0

    for folder, collection_name in COLLECTION_MAP.items():
        collection = client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"},
        )
        approved = {rel for rel, e in docs.items()
                    if rel.startswith(f"{folder}/")
                    and e.get("status") in ("approved", "indexed")
                    and (seed_dir / rel).exists()}
        for rel in sorted(approved):
            added, pruned = upsert_document(collection, seed_dir / rel, embed_fn)
            total_added += added
            total_pruned += pruned
            text = (seed_dir / rel).read_text(encoding="utf-8")
            M.update_entry(manifest, rel, status="indexed",
                           sha256=M.compute_sha256(text))
            print(f"[{collection_name}] {rel}: +{added} chunks, -{pruned} podados")
        total_pruned += prune_orphan_sources(
            collection, {Path(rel).name for rel in approved})

    M.save_manifest(manifest, manifest_path)
    print(f"\nIndexado: +{total_added} chunks, -{total_pruned} podados.")
    return 0


def _real_embed_fn():
    """Embeddings Gemini vía get_model('embedding') — solo camino CLI."""
    from google import genai
    from config.model_loader import get_model

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("ERROR: GEMINI_API_KEY no definida.")
    client = genai.Client(api_key=api_key)
    model = get_model("embedding")
    print(f"Modelo de embedding: {model}")

    def embed(texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            res = client.models.embed_content(model=model, contents=[t])
            out.append(res.embeddings[0].values)
            time.sleep(RATE_LIMIT_SLEEP)
        return out

    return embed


def _real_chroma_client():
    import chromadb
    root = Path(__file__).parent.parent
    chroma_dir = Path(os.environ.get("CHROMA_DB_PATH", str(root / "knowledge" / "chroma_db")))
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(chroma_dir))


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Indexa el corpus aprobado en ChromaDB")
    p.add_argument("--seed-dir", default="data/kb_seed")
    p.add_argument("--manifest", default=str(M.DEFAULT_PATH))
    return p.parse_args()


if __name__ == "__main__":
    a = _args()
    raise SystemExit(run(Path(a.seed_dir), Path(a.manifest),
                         _real_chroma_client(), _real_embed_fn()))
