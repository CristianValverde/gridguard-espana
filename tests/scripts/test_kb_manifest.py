import json
from pathlib import Path

import pytest

from scripts import kb_manifest as M


def test_update_entry_creates_with_generated_at(tmp_path):
    manifest = M.load_manifest(tmp_path / "manifest.json")  # no existe → vacío
    entry = M.update_entry(manifest, "procedures/x.txt", status="draft",
                           sha256="abc", source_urls=["https://ree.es"])
    assert entry["status"] == "draft"
    assert entry["sha256"] == "abc"
    assert entry["source_urls"] == ["https://ree.es"]
    assert "generated_at" in entry


def test_status_transitions_and_invalid_status_rejected():
    manifest = {"documents": {}}
    M.update_entry(manifest, "a.txt", status="draft")
    M.update_entry(manifest, "a.txt", status="validated")
    M.update_entry(manifest, "a.txt", status="approved", reviewed_by="usuario")
    assert M.get_status(manifest, "a.txt") == "approved"
    assert manifest["documents"]["a.txt"]["reviewed_by"] == "usuario"
    with pytest.raises(ValueError):
        M.update_entry(manifest, "a.txt", status="publicado")


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = {"documents": {"b.txt": {"status": "indexed", "sha256": "x"}}}
    M.save_manifest(manifest, path)
    assert M.load_manifest(path) == manifest
    # JSON legible (indentado, utf-8)
    assert "\n" in path.read_text(encoding="utf-8")


def test_compute_sha256_is_stable():
    assert M.compute_sha256("hola") == M.compute_sha256("hola")
    assert M.compute_sha256("hola") != M.compute_sha256("adios")


def test_bootstrap_registers_existing_txt_as_approved(tmp_path):
    seed = tmp_path / "kb_seed"
    (seed / "procedures").mkdir(parents=True)
    (seed / "procedures" / "legacy.txt").write_text("contenido", encoding="utf-8")
    mpath = tmp_path / "manifest.json"
    n = M.bootstrap(seed, mpath, reviewed_by="bootstrap-test")
    assert n == 1
    manifest = M.load_manifest(mpath)
    entry = manifest["documents"]["procedures/legacy.txt"]
    assert entry["status"] == "approved"
    assert entry["reviewed_by"] == "bootstrap-test"
    assert entry["sha256"] == M.compute_sha256("contenido")
    # idempotente: segunda pasada no re-registra
    assert M.bootstrap(seed, mpath, reviewed_by="otro") == 0
