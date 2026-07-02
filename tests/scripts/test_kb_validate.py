from pathlib import Path

import pytest

from scripts import kb_manifest as M
from scripts import kb_validate as V

_GOOD = """PROCEDIMIENTO DE PRUEBA
La interconexión España-Francia tiene una capacidad comercial de 2.800 MW.
La interconexión ES-PT alcanza una capacidad de 3000 MW (MIBEL).

FUENTES
- https://www.ree.es/es
"""

_BAD_CAPACITY = """PROCEDIMIENTO ERRÓNEO
La interconexión ES-PT tiene una capacidad de 1400 MW.

FUENTES
- https://example.org
"""

_UNVERIFIED = """DOCUMENTO INCOMPLETO
El umbral es [SIN VERIFICAR] MW según el PO.

FUENTES
- https://example.org
"""

_NO_SOURCES = "DOCUMENTO SIN FUENTES\nTexto cualquiera.\n"


def test_good_document_passes(tmp_path):
    p = tmp_path / "ok.txt"
    p.write_text(_GOOD, encoding="utf-8")
    assert V.validate_document(p) == []


def test_wrong_capacity_flagged():
    errors = V.check_capacity_consistency(_BAD_CAPACITY)
    assert len(errors) == 1
    assert "es_pt" in errors[0] and "1400" in errors[0]


def test_unverified_marker_flagged_with_line_number():
    lines = V.find_unverified_markers(_UNVERIFIED)
    assert lines == [2]


def test_missing_sources_section_flagged(tmp_path):
    p = tmp_path / "nosrc.txt"
    p.write_text(_NO_SOURCES, encoding="utf-8")
    errors = V.validate_document(p)
    assert any("FUENTES" in e for e in errors)


def test_allowed_capacities_cover_canonical_ntc():
    """Coherencia con la fuente única: los NTC canónicos deben estar permitidos."""
    from ml.features import INTERCONNECT_CAPACITY_MW
    for link, cap in INTERCONNECT_CAPACITY_MW.items():
        assert int(cap) in V.ALLOWED_CAPACITY_MW[link], link


def test_run_promotes_draft_to_validated(tmp_path):
    seed = tmp_path / "kb_seed"
    (seed / "procedures").mkdir(parents=True)
    (seed / "procedures" / "doc.txt").write_text(_GOOD, encoding="utf-8")
    mpath = tmp_path / "manifest.json"
    manifest = M.load_manifest(mpath)
    M.update_entry(manifest, "procedures/doc.txt", status="draft",
                   sha256=M.compute_sha256(_GOOD))
    M.save_manifest(manifest, mpath)

    rc = V.run(seed, mpath)
    assert rc == 0
    assert M.get_status(M.load_manifest(mpath), "procedures/doc.txt") == "validated"


def test_run_fails_and_keeps_draft_on_bad_doc(tmp_path):
    seed = tmp_path / "kb_seed"
    (seed / "procedures").mkdir(parents=True)
    (seed / "procedures" / "malo.txt").write_text(_BAD_CAPACITY, encoding="utf-8")
    mpath = tmp_path / "manifest.json"
    manifest = M.load_manifest(mpath)
    M.update_entry(manifest, "procedures/malo.txt", status="draft",
                   sha256=M.compute_sha256(_BAD_CAPACITY))
    M.save_manifest(manifest, mpath)

    rc = V.run(seed, mpath)
    assert rc != 0
    assert M.get_status(M.load_manifest(mpath), "procedures/malo.txt") == "draft"
