from pathlib import Path

import pytest

chromadb = pytest.importorskip("chromadb")

from scripts import kb_embed as E
from scripts import kb_manifest as M


def _fake_embed(texts):
    # determinista, sin red: 8 dims función de la longitud
    return [[float((len(t) + i) % 7) for i in range(8)] for t in texts]


@pytest.fixture()
def collection():
    client = chromadb.EphemeralClient()
    return client.get_or_create_collection("test_proc", metadata={"hnsw:space": "cosine"})


def test_chunk_ids_are_content_addressed():
    chunks_a = E.chunk_text("x" * 1500, "doc.txt")
    chunks_b = E.chunk_text("x" * 1500, "doc.txt")
    assert [c["id"] for c in chunks_a] == [c["id"] for c in chunks_b]  # estable
    chunks_c = E.chunk_text("y" * 1500, "doc.txt")
    assert chunks_a[0]["id"] != chunks_c[0]["id"]  # cambia con el contenido


def test_upsert_is_idempotent(tmp_path, collection):
    doc = tmp_path / "doc.txt"
    doc.write_text("contenido de prueba " * 100, encoding="utf-8")
    added1, pruned1 = E.upsert_document(collection, doc, _fake_embed)
    assert added1 > 0 and pruned1 == 0
    added2, pruned2 = E.upsert_document(collection, doc, _fake_embed)
    assert (added2, pruned2) == (0, 0)
    assert collection.count() == added1


def test_upsert_reindexes_changed_content_and_prunes_stale(tmp_path, collection):
    doc = tmp_path / "doc.txt"
    doc.write_text("version uno " * 100, encoding="utf-8")
    added1, _ = E.upsert_document(collection, doc, _fake_embed)
    doc.write_text("version dos distinta " * 100, encoding="utf-8")
    added2, pruned2 = E.upsert_document(collection, doc, _fake_embed)
    assert added2 > 0 and pruned2 == added1  # lo viejo se poda


def test_prune_orphan_sources(tmp_path, collection):
    doc = tmp_path / "huerfano.txt"
    doc.write_text("texto " * 100, encoding="utf-8")
    E.upsert_document(collection, doc, _fake_embed)
    n = E.prune_orphan_sources(collection, approved_names=set())
    assert n > 0 and collection.count() == 0


def test_run_only_indexes_approved(tmp_path):
    seed = tmp_path / "kb_seed"
    (seed / "procedures").mkdir(parents=True)
    (seed / "procedures" / "ok.txt").write_text("aprobado " * 100, encoding="utf-8")
    (seed / "procedures" / "draft.txt").write_text("borrador " * 100, encoding="utf-8")
    mpath = tmp_path / "manifest.json"
    manifest = M.load_manifest(mpath)
    M.update_entry(manifest, "procedures/ok.txt", status="approved")
    M.update_entry(manifest, "procedures/draft.txt", status="draft")
    M.save_manifest(manifest, mpath)

    client = chromadb.EphemeralClient()
    rc = E.run(seed, mpath, client, _fake_embed)
    assert rc == 0
    col = client.get_or_create_collection("procedures")
    sources = {m["source"] for m in col.get(include=["metadatas"])["metadatas"]}
    assert sources == {"ok.txt"}
    assert M.get_status(M.load_manifest(mpath), "procedures/ok.txt") == "indexed"
    assert M.get_status(M.load_manifest(mpath), "procedures/draft.txt") == "draft"


def test_embed_with_retry_recovers_from_transient_failure():
    calls = {"n": 0}

    def flaky(texts):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return _fake_embed(texts)

    out = E._embed_with_retry(flaky, ["a", "b"], attempts=3, base_sleep=0.0)
    assert len(out) == 2 and calls["n"] == 2
