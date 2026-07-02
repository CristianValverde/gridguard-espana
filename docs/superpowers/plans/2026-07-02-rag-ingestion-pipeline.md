# Pipeline build-time de ingesta RAG · Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pipeline offline de dos componentes (investigador ADK híbrido + embedder determinista) que enriquece el corpus RAG con trazabilidad, gate humano e indexado idempotente.

**Architecture:** `topics.yaml → researcher (ADK+google_search) → .txt draft → kb_validate (Python puro) → validated → aprobación humana → approved → kb_embed (Python puro) → ChromaDB (indexed)`. Manifiesto `data/kb_seed/manifest.json` como fuente de verdad del estado. Runtime sin cambios.

**Tech Stack:** Python 3.14, google-adk 2.3.0 (`as_static_instruction`), google-genai (embeddings), ChromaDB, PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-07-02-rag-ingestion-agents-design.md`

## Global Constraints

- NUNCA leer `.env`, `*.key`, `secrets.*`, `credentials.*`. Tokens vía variables de entorno.
- NUNCA hardcodear nombres de modelos Gemini — `config/model_loader.py::get_model()`.
- Shift-left: el único componente con LLM es el investigador.
- Suite offline por defecto VERDE (`pytest -q`; `pytest.ini`: `addopts = -m "not integration"`, `testpaths = eval tests`). Lo que requiera Gemini/Chroma real → `@pytest.mark.integration`.
- Commits en español, conventional, terminados en `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Tareas [FRÍO] verificables offline; la Task 5 es [LIVE] (Gemini API + gate humano).
- `api-probe` OBLIGATORIO antes de escribir código de integración con cualquier fuente/API
  externa NUEVA (el tool built-in `google_search` de ADK no lo requiere: no hay código de
  integración propio).

## File Structure

- Create `scripts/kb_manifest.py` — estado del corpus (load/save/update/bootstrap). Sin I/O de red.
- Create `scripts/kb_validate.py` — checks deterministas + promoción draft→validated.
- Create `scripts/kb_embed.py` — chunk/hash/upsert idempotente + poda; reemplaza `scripts/seed_knowledge_base.py`.
- Create `data/kb_seed/topics.yaml` — cola de temas versionada.
- Create `agents/ingest/__init__.py`, `agents/ingest/researcher.py`, `agents/ingest/run_researcher.py`.
- Delete `scripts/seed_knowledge_base.py` (Task 3, tras absorber su lógica).
- Tests: `tests/scripts/test_kb_manifest.py`, `tests/scripts/test_kb_validate.py`, `tests/scripts/test_kb_embed.py`, `tests/agents/test_ingest_researcher.py`.

---

### Task 1: `scripts/kb_manifest.py` — manifiesto de trazabilidad  [FRÍO]

**Files:**
- Create: `scripts/kb_manifest.py`
- Test: `tests/scripts/test_kb_manifest.py`

**Interfaces:**
- Produces (consumidas por Tasks 2, 3, 4, 5):
  - `STATUSES = ("draft", "validated", "approved", "indexed")`
  - `DEFAULT_PATH = Path("data/kb_seed/manifest.json")`
  - `compute_sha256(text: str) -> str`
  - `load_manifest(path: Path = DEFAULT_PATH) -> dict` (estructura `{"documents": {rel_file: entry}}`)
  - `save_manifest(manifest: dict, path: Path = DEFAULT_PATH) -> None`
  - `update_entry(manifest, rel_file, *, status=None, sha256=None, source_urls=None, reviewed_by=None) -> dict` (crea la entrada con `generated_at` UTC si no existe)
  - `get_status(manifest, rel_file) -> str | None`
  - `bootstrap(seed_dir: Path, manifest_path: Path, reviewed_by: str) -> int` — registra como `approved` los `.txt` existentes sin entrada (corpus legacy ya reconciliado a mano).

- [ ] **Step 1: Escribir los tests (fallan)**

```python
# tests/scripts/test_kb_manifest.py
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
```

- [ ] **Step 2: Ejecutar y ver que falla**

Run: `python -m pytest tests/scripts/test_kb_manifest.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'scripts.kb_manifest'`).
Nota: si `tests/scripts/` no existe aún, crearlo (sin `__init__.py`, como el resto de `tests/`).

- [ ] **Step 3: Implementar `scripts/kb_manifest.py`**

```python
"""scripts/kb_manifest.py — Manifiesto de trazabilidad del corpus RAG.

Fuente de verdad del estado de cada documento de data/kb_seed/:
  draft → validated (kb_validate) → approved (humano) → indexed (kb_embed).
Estructura: {"documents": {"<carpeta>/<archivo>.txt": {status, sha256,
source_urls, generated_at, reviewed_by}}}. Python puro, sin red.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

STATUSES = ("draft", "validated", "approved", "indexed")
DEFAULT_PATH = Path("data/kb_seed/manifest.json")


def compute_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_manifest(path: Path = DEFAULT_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        return {"documents": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(manifest: dict, path: Path = DEFAULT_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def update_entry(manifest: dict, rel_file: str, *, status: str | None = None,
                 sha256: str | None = None, source_urls: list[str] | None = None,
                 reviewed_by: str | None = None) -> dict:
    if status is not None and status not in STATUSES:
        raise ValueError(f"status inválido: {status!r} (válidos: {STATUSES})")
    docs = manifest.setdefault("documents", {})
    entry = docs.setdefault(rel_file, {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    if status is not None:
        entry["status"] = status
    if sha256 is not None:
        entry["sha256"] = sha256
    if source_urls is not None:
        entry["source_urls"] = source_urls
    if reviewed_by is not None:
        entry["reviewed_by"] = reviewed_by
    return entry


def get_status(manifest: dict, rel_file: str) -> str | None:
    return manifest.get("documents", {}).get(rel_file, {}).get("status")


def bootstrap(seed_dir: Path, manifest_path: Path = DEFAULT_PATH,
              reviewed_by: str = "bootstrap") -> int:
    """Registra como `approved` los .txt existentes sin entrada en el manifiesto.

    Para el corpus legacy (escrito y reconciliado a mano antes del pipeline).
    Idempotente: los archivos ya registrados no se tocan.
    """
    seed_dir = Path(seed_dir)
    manifest = load_manifest(manifest_path)
    added = 0
    for txt in sorted(seed_dir.glob("*/*.txt")):
        rel = txt.relative_to(seed_dir).as_posix()
        if rel in manifest.get("documents", {}):
            continue
        text = txt.read_text(encoding="utf-8")
        update_entry(manifest, rel, status="approved",
                     sha256=compute_sha256(text), reviewed_by=reviewed_by)
        added += 1
    if added:
        save_manifest(manifest, manifest_path)
    return added
```

- [ ] **Step 4: Ejecutar y ver que pasa**

Run: `python -m pytest tests/scripts/test_kb_manifest.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/kb_manifest.py tests/scripts/test_kb_manifest.py
git commit -m "feat(kb): manifiesto de trazabilidad del corpus (draft->validated->approved->indexed)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `scripts/kb_validate.py` — validador determinista  [FRÍO]

**Files:**
- Create: `scripts/kb_validate.py`
- Test: `tests/scripts/test_kb_validate.py`

**Interfaces:**
- Consumes: `scripts.kb_manifest` (load/save/update_entry/get_status/compute_sha256), `ml.features.INTERCONNECT_CAPACITY_MW`.
- Produces (consumidas por Task 5):
  - `UNVERIFIED_MARKER = "[SIN VERIFICAR]"`
  - `ALLOWED_CAPACITY_MW: dict[str, set[int]]`
  - `find_unverified_markers(text: str) -> list[int]` (números de línea, 1-based)
  - `has_sources_section(text: str) -> bool`
  - `check_capacity_consistency(text: str) -> list[str]` (descripciones de violación)
  - `validate_document(path: Path) -> list[str]` (vacío = OK)
  - `run(seed_dir: Path, manifest_path: Path, files: list[str] | None = None) -> int` (exit code; promueve draft→validated)
  - CLI: `python -m scripts.kb_validate [--seed-dir D] [--manifest M] [--file rel.txt ...]`

- [ ] **Step 1: Escribir los tests (fallan)**

```python
# tests/scripts/test_kb_validate.py
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
```

- [ ] **Step 2: Ejecutar y ver que falla**

Run: `python -m pytest tests/scripts/test_kb_validate.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'scripts.kb_validate'`).

- [ ] **Step 3: Implementar `scripts/kb_validate.py`**

```python
"""scripts/kb_validate.py — Gate determinista del corpus RAG (sin LLM).

Checks por documento:
  1. Sin marcadores [SIN VERIFICAR] pendientes.
  2. Sección FUENTES presente.
  3. Consistencia de capacidades de interconexión: toda línea que mencione un
     enlace Y afirme "capacidad ... N MW" debe usar un valor de la allowlist
     (canónicos de ml.features.INTERCONNECT_CAPACITY_MW + valores documentados).
Promueve draft→validated en el manifiesto; un fallo deja el doc en draft.

Uso: python -m scripts.kb_validate [--seed-dir data/kb_seed] [--manifest ...]
                                   [--file procedures/x.txt ...]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.features import INTERCONNECT_CAPACITY_MW
from scripts import kb_manifest as M

UNVERIFIED_MARKER = "[SIN VERIFICAR]"

# Allowlist por enlace: canónico (ml.features) + valores legítimos en contexto
# (600 = import MA→ES; 1400 = capacidad instalada total MA e histórico ES-FR
# pre-INELFE; ambos aparecen en el corpus reconciliado en Task 11 de eficacia).
ALLOWED_CAPACITY_MW: dict[str, set[int]] = {
    "es_fr": {int(INTERCONNECT_CAPACITY_MW["es_fr"]), 1400, 5000},
    "es_pt": {int(INTERCONNECT_CAPACITY_MW["es_pt"])},
    "es_ma": {int(INTERCONNECT_CAPACITY_MW["es_ma"]), 600, 1400},
}

_LINK_RE = {
    "es_fr": re.compile(r"ES[-–]FR|Espa\w+[-–\s]+Franc|Franc\w+[-–\s]+Espa", re.IGNORECASE),
    "es_pt": re.compile(r"ES[-–]PT|Espa\w+[-–\s]+Portug|Portug\w+[-–\s]+Espa", re.IGNORECASE),
    "es_ma": re.compile(r"ES[-–]MA|Espa\w+[-–\s]+Marruec|Marruec\w+[-–\s]+Espa", re.IGNORECASE),
}
_CAPACITY_RE = re.compile(r"capacidad[^.\n]*?(\d{1,2}[.,]\d{3}|\d{3,5})\s*MW", re.IGNORECASE)


def find_unverified_markers(text: str) -> list[int]:
    return [i for i, line in enumerate(text.splitlines(), start=1)
            if UNVERIFIED_MARKER in line]


def has_sources_section(text: str) -> bool:
    return re.search(r"^FUENTES\b", text, re.MULTILINE) is not None


def _parse_mw(raw: str) -> int:
    return int(raw.replace(".", "").replace(",", ""))


def check_capacity_consistency(text: str) -> list[str]:
    errors: list[str] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        caps = [_parse_mw(m.group(1)) for m in _CAPACITY_RE.finditer(line)]
        if not caps:
            continue
        for link, link_re in _LINK_RE.items():
            if not link_re.search(line):
                continue
            for value in caps:
                if value not in ALLOWED_CAPACITY_MW[link]:
                    errors.append(
                        f"línea {lineno}: capacidad {value} MW para {link} no está "
                        f"en la allowlist {sorted(ALLOWED_CAPACITY_MW[link])}"
                    )
    return errors


def validate_document(path: Path) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    errors: list[str] = []
    for lineno in find_unverified_markers(text):
        errors.append(f"línea {lineno}: marcador {UNVERIFIED_MARKER} pendiente")
    if not has_sources_section(text):
        errors.append("falta la sección FUENTES")
    errors.extend(check_capacity_consistency(text))
    return errors


def run(seed_dir: Path = Path("data/kb_seed"),
        manifest_path: Path = M.DEFAULT_PATH,
        files: list[str] | None = None) -> int:
    seed_dir = Path(seed_dir)
    manifest = M.load_manifest(manifest_path)
    if files is None:
        files = [rel for rel, e in manifest.get("documents", {}).items()
                 if e.get("status") == "draft"]
    if not files:
        print("Sin documentos en draft que validar.")
        return 0

    failures = 0
    for rel in files:
        path = seed_dir / rel
        if not path.exists():
            print(f"[FAIL] {rel}: archivo no encontrado")
            failures += 1
            continue
        errors = validate_document(path)
        if errors:
            failures += 1
            print(f"[FAIL] {rel}:")
            for e in errors:
                print(f"       - {e}")
        else:
            M.update_entry(manifest, rel, status="validated",
                           sha256=M.compute_sha256(path.read_text(encoding="utf-8")))
            print(f"[OK]   {rel} -> validated")
    M.save_manifest(manifest, manifest_path)
    print(f"\nValidación: {len(files) - failures}/{len(files)} documentos OK.")
    return 1 if failures else 0


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Valida documentos del corpus RAG")
    p.add_argument("--seed-dir", default="data/kb_seed")
    p.add_argument("--manifest", default=str(M.DEFAULT_PATH))
    p.add_argument("--file", action="append", dest="files", default=None,
                   help="ruta relativa (repetible); default: todos los draft")
    return p.parse_args()


if __name__ == "__main__":
    a = _args()
    raise SystemExit(run(Path(a.seed_dir), Path(a.manifest), a.files))
```

- [ ] **Step 4: Ejecutar y ver que pasa**

Run: `python -m pytest tests/scripts/test_kb_validate.py -q`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/kb_validate.py tests/scripts/test_kb_validate.py
git commit -m "feat(kb): validador determinista del corpus (capacidades, FUENTES, SIN VERIFICAR)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `scripts/kb_embed.py` — embedder idempotente (reemplaza seed)  [FRÍO]

**Files:**
- Create: `scripts/kb_embed.py`
- Delete: `scripts/seed_knowledge_base.py` (Step 6)
- Modify: `tests/mcp_servers/test_kb_interconnexiones.py:4-5` (docstring: `scripts.seed_knowledge_base` → `scripts.kb_embed`)
- Test: `tests/scripts/test_kb_embed.py`

**Interfaces:**
- Consumes: `scripts.kb_manifest`, `config.model_loader.get_model("embedding")` (solo en CLI, no en tests).
- Produces (consumidas por Task 5):
  - `COLLECTION_MAP = {"regulations": "regulations", "incidents": "incidents", "procedures": "procedures"}`
  - `CHUNK_SIZE = 600`, `CHUNK_OVERLAP = 100`
  - `chunk_text(text: str, source_name: str) -> list[dict]` (dicts con `id`, `text`, `source`, `chunk_index`; `id = f"{stem}_c{idx:03d}_{sha8}"`)
  - `upsert_document(collection, txt_path: Path, embed_fn) -> tuple[int, int]` (añadidos, podados)
  - `prune_orphan_sources(collection, approved_names: set[str]) -> int`
  - `run(seed_dir, manifest_path, client, embed_fn) -> int` (indexa `approved`/`indexed`, promueve a `indexed`)
  - CLI: `python -m scripts.kb_embed` (construye cliente Chroma real + embeddings Gemini)

- [ ] **Step 1: Escribir los tests (fallan)** — ChromaDB efímero + embeddings fake, todo offline

```python
# tests/scripts/test_kb_embed.py
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
```

- [ ] **Step 2: Ejecutar y ver que falla**

Run: `python -m pytest tests/scripts/test_kb_embed.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'scripts.kb_embed'`).

- [ ] **Step 3: Implementar `scripts/kb_embed.py`**

```python
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
```

- [ ] **Step 4: Ejecutar y ver que pasa**

Run: `python -m pytest tests/scripts/test_kb_embed.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Actualizar la referencia en el smoke test**

En `tests/mcp_servers/test_kb_interconnexiones.py`, docstring línea 4: sustituir
`python -m scripts.seed_knowledge_base` por `python -m scripts.kb_embed`.

- [ ] **Step 6: Retirar el script antiguo y verificar que nada lo importa**

```bash
git rm scripts/seed_knowledge_base.py
grep -rn "seed_knowledge_base" --include="*.py" .
```
Expected: sin resultados en código (solo menciones en docs/ y .superpowers/, que son registro histórico).

- [ ] **Step 7: Suite completa y commit**

Run: `python -m pytest -q`
Expected: verde (los nuevos tests incluidos; sin regresiones).

```bash
git add scripts/kb_embed.py tests/scripts/test_kb_embed.py tests/mcp_servers/test_kb_interconnexiones.py
git commit -m "feat(kb): embedder idempotente con poda y manifiesto; retira seed_knowledge_base

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Agente investigador ADK + cola de temas  [FRÍO]

**Files:**
- Create: `data/kb_seed/topics.yaml`
- Create: `agents/ingest/__init__.py` (vacío)
- Create: `agents/ingest/researcher.py`
- Create: `agents/ingest/run_researcher.py`
- Test: `tests/agents/test_ingest_researcher.py`

**Interfaces:**
- Consumes: `config.model_loader.get_model("smart_agent")`, `agents.instruction_utils.as_static_instruction`, `google.adk.tools.google_search`, `scripts.kb_manifest`.
- Produces:
  - `agents.ingest.researcher.build_researcher() -> google.adk.agents.Agent`
  - `agents.ingest.researcher._INSTRUCTION: str`
  - `agents.ingest.run_researcher.load_topics(path) -> list[dict]` (dicts `{id, collection, title, status}`)
  - `agents.ingest.run_researcher.run_topic(topic: dict, seed_dir, manifest_path) -> Path` (escribe el .txt draft)
  - CLI: `python -m agents.ingest.run_researcher --topic <id>` | `--all-pending`

- [ ] **Step 1: Escribir los tests (fallan)** — contrato offline, sin LLM vivo

```python
# tests/agents/test_ingest_researcher.py
from pathlib import Path

import pytest
import yaml


def test_topics_yaml_parses_with_required_fields():
    topics = yaml.safe_load(Path("data/kb_seed/topics.yaml").read_text(encoding="utf-8"))
    assert isinstance(topics, list) and topics
    for t in topics:
        assert {"id", "collection", "title", "status"} <= set(t.keys())
        assert t["collection"] in ("regulations", "procedures", "incidents")
        assert t["status"] in ("pending", "generated")


def test_researcher_builds_with_model_from_registry():
    from agents.ingest.researcher import build_researcher
    from config.model_loader import get_model

    agent = build_researcher()
    assert agent.name == "kb_researcher"
    assert agent.model == get_model("smart_agent")  # nunca hardcodeado


def test_instruction_contract():
    """Las reglas críticas del diseño deben estar en la instrucción."""
    from agents.ingest.researcher import _INSTRUCTION

    assert "[SIN VERIFICAR]" in _INSTRUCTION
    assert "FUENTES" in _INSTRUCTION
    assert "google_search" in _INSTRUCTION or "búsqueda" in _INSTRUCTION.lower()
    assert "MAYÚSCULAS" in _INSTRUCTION


def test_extract_text_takes_last_text_event():
    from agents.ingest.run_researcher import _extract_text

    class _Part:
        def __init__(self, text):
            self.text = text

    class _Content:
        def __init__(self, text):
            self.parts = [_Part(text)]

    class _Event:
        def __init__(self, text):
            self.content = _Content(text) if text else None

    events = [_Event("intermedio"), _Event(None), _Event("DOCUMENTO FINAL\n\nFUENTES\n- x")]
    assert _extract_text(events).startswith("DOCUMENTO FINAL")
```

- [ ] **Step 2: Ejecutar y ver que falla**

Run: `python -m pytest tests/agents/test_ingest_researcher.py -q`
Expected: FAIL (topics.yaml y módulos no existen).

- [ ] **Step 3: Crear `data/kb_seed/topics.yaml`**

```yaml
# Cola de temas del agente investigador (agents/ingest/run_researcher.py).
# status: pending → generated (el .txt draft existe; su estado vive en manifest.json)
- id: po_operacion_ree
  collection: regulations
  title: >
    Procedimientos de Operación de REE relevantes para estabilidad: PO 1.1/1.2
    (funcionamiento y seguridad), PO 3.1 (programación), PO 7.1-7.4 (reservas y
    regulación), PO 10.1 (medidas). Qué regula cada uno y umbrales operativos clave.
  status: pending
- id: entsoe_emergency_restoration
  collection: regulations
  title: >
    Network Code de ENTSO-E on Emergency and Restoration (NC ER) y SO GL:
    estados del sistema (normal/alerta/emergencia/blackout/restauración),
    obligaciones de los TSO y coordinación transfronteriza.
  status: pending
- id: procedimientos_deslastre_reposicion
  collection: procedures
  title: >
    Deslastre de cargas por subfrecuencia en España (escalones, umbrales Hz) y
    plan de reposición del servicio (arranque autógeno, islas eléctricas,
    resincronización con Europa).
  status: pending
- id: post_mortem_apagon_2025
  collection: incidents
  title: >
    Post-mortem del apagón ibérico del 28 de abril de 2025: cronología
    minuto a minuto, causas (oscilaciones, baja inercia sincrónica, cascada),
    papel de las interconexiones y lecciones aprendidas del informe de REE.
  status: pending
- id: separacion_iberica_2006
  collection: incidents
  title: >
    Incidente europeo del 4 de noviembre de 2006: separación del sistema UCTE
    en tres islas, efecto en la península ibérica, y comparación con 2025.
  status: pending
- id: acuerdos_intercambio_fr_pt_ma
  collection: procedures
  title: >
    Acuerdos y mecanismos de intercambio de España con Francia (BALIT, subastas
    INELFE), Portugal (MIBEL, coupling intradiario) y Marruecos (contrato ONEE):
    cómo se programan y activan los intercambios de emergencia.
  status: pending
```

- [ ] **Step 4: Crear `agents/ingest/__init__.py` y `agents/ingest/researcher.py`**

`agents/ingest/__init__.py`: archivo vacío.

```python
"""agents/ingest/researcher.py — Agente investigador del corpus RAG (build-time).

Genera documentos .txt de dominio para data/kb_seed/ en modo HÍBRIDO: redacta
por síntesis pero verifica todo número operativo con google_search antes de
escribirlo; si no puede verificarlo, deja el marcador [SIN VERIFICAR] (el
validador bloqueará el documento hasta resolverlo). NO es un agente de runtime.
"""
from __future__ import annotations

from google.adk.agents import Agent
from google.adk.tools import google_search
from google.genai import types as genai_types

from agents.instruction_utils import as_static_instruction
from config.model_loader import get_model

_INSTRUCTION = """
Eres el investigador de dominio de GridGuard España. Escribes documentos de
conocimiento sobre la red eléctrica española para una base RAG.

FORMATO (obligatorio, idéntico a los .txt existentes de data/kb_seed/):
- Texto plano en español. SIN markdown.
- Título y encabezados de sección en MAYÚSCULAS.
- 800-2500 palabras, denso en datos operativos.
- Última sección: FUENTES — lista de URLs/referencias oficiales usadas.

VERIFICACIÓN (regla crítica):
- TODO número operativo (MW de capacidad, umbrales en Hz, fechas, duraciones,
  escalones de deslastre) debe verificarse con la herramienta google_search
  contra fuentes oficiales (ree.es, entsoe.eu, BOE, CNMC) ANTES de escribirlo.
- Si no logras verificar un dato, escribe exactamente [SIN VERIFICAR] en su
  lugar. NUNCA inventes un número, un ID de procedimiento (PO) ni una fecha.
- Capacidades de interconexión canónicas del proyecto: ES-FR 2800 MW,
  ES-PT 3000 MW, ES-MA 900 MW export / 600 MW import (AC submarino
  Tarifa-Fardioua, NO es HVDC). Si tu investigación contradice estos valores,
  usa [SIN VERIFICAR] y anótalo en FUENTES.

SALIDA: devuelve ÚNICAMENTE el texto plano del documento, sin comentarios.
""".strip()


def build_researcher() -> Agent:
    return Agent(
        model=get_model("smart_agent"),
        name="kb_researcher",
        instruction=as_static_instruction(_INSTRUCTION),
        tools=[google_search],
        generate_content_config=genai_types.GenerateContentConfig(temperature=0.3),
    )
```

- [ ] **Step 5: Crear `agents/ingest/run_researcher.py`**

```python
"""agents/ingest/run_researcher.py — Runner offline del investigador.

Uso:
  python -m agents.ingest.run_researcher --topic po_operacion_ree
  python -m agents.ingest.run_researcher --all-pending

Por tema: ejecuta el agente, escribe data/kb_seed/<collection>/<id>.txt,
registra la entrada draft en el manifiesto y marca el tema como generated.
Siguiente paso del pipeline: python -m scripts.kb_validate
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from scripts import kb_manifest as M

TOPICS_PATH = Path("data/kb_seed/topics.yaml")
SEED_DIR = Path("data/kb_seed")


def load_topics(path: Path = TOPICS_PATH) -> list[dict]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def save_topics(topics: list[dict], path: Path = TOPICS_PATH) -> None:
    Path(path).write_text(
        yaml.safe_dump(topics, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _extract_text(events) -> str:
    """Último evento con contenido textual = documento final del agente."""
    for event in reversed(list(events)):
        content = getattr(event, "content", None)
        if content and getattr(content, "parts", None):
            texts = [p.text for p in content.parts if getattr(p, "text", None)]
            if texts:
                return "\n".join(texts).strip()
    raise RuntimeError("El agente no devolvió texto")


def run_topic(topic: dict, seed_dir: Path = SEED_DIR,
              manifest_path: Path = M.DEFAULT_PATH) -> Path:
    from agents.ingest.researcher import build_researcher

    session_service = InMemorySessionService()
    runner = Runner(agent=build_researcher(), app_name="gridguard_ingest",
                    session_service=session_service)
    # ADK 2.3.0: create_session es corutina — await vía asyncio.run.
    session = asyncio.run(session_service.create_session(
        app_name="gridguard_ingest", user_id="ingest-runner"))

    prompt = (
        f"Escribe el documento para la colección '{topic['collection']}'.\n"
        f"TEMA: {topic['title']}"
    )
    message = genai_types.Content(role="user",
                                  parts=[genai_types.Part(text=prompt)])
    events = list(runner.run(session_id=session.id, user_id="ingest-runner",
                             new_message=message))
    text = _extract_text(events)

    out_path = seed_dir / topic["collection"] / f"{topic['id']}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n", encoding="utf-8")

    manifest = M.load_manifest(manifest_path)
    rel = out_path.relative_to(seed_dir).as_posix()
    M.update_entry(manifest, rel, status="draft", sha256=M.compute_sha256(text + "\n"))
    M.save_manifest(manifest, manifest_path)
    print(f"[draft] {rel} ({len(text.split())} palabras)")
    return out_path


def main() -> int:
    p = argparse.ArgumentParser(description="Genera documentos del corpus RAG")
    p.add_argument("--topic", help="id del tema en topics.yaml")
    p.add_argument("--all-pending", action="store_true")
    a = p.parse_args()

    topics = load_topics()
    if a.topic:
        selected = [t for t in topics if t["id"] == a.topic]
        if not selected:
            sys.exit(f"Tema no encontrado: {a.topic}")
    elif a.all_pending:
        selected = [t for t in topics if t["status"] == "pending"]
    else:
        sys.exit("Indica --topic <id> o --all-pending")

    for topic in selected:
        run_topic(topic)
        topic["status"] = "generated"
        save_topics(topics)
    print(f"\n{len(selected)} documento(s) en draft. "
          f"Siguiente: python -m scripts.kb_validate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Ejecutar y ver que pasa**

Run: `python -m pytest tests/agents/test_ingest_researcher.py -q`
Expected: PASS (4 tests).

- [ ] **Step 7: Suite completa y commit**

Run: `python -m pytest -q`
Expected: verde.

```bash
git add data/kb_seed/topics.yaml agents/ingest/ tests/agents/test_ingest_researcher.py
git commit -m "feat(agents): investigador ADK del corpus RAG (hibrido, offline) + cola de temas

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Ejecución del pipeline real + smoke de recuperación  [LIVE — Gemini API + gate humano]

**Files:**
- Create: `data/kb_seed/manifest.json` (generado por bootstrap)
- Modify: `data/kb_seed/topics.yaml` (status → generated)
- (Generados): `.txt` nuevos en `data/kb_seed/*/`

**Interfaces:**
- Consumes: todo lo anterior. Requiere `GEMINI_API_KEY` en el entorno (nunca leer `.env` directamente).

- [ ] **Step 1: Bootstrap del corpus legacy** (los .txt existentes, ya reconciliados a mano, entran como approved)

```bash
python -c "from scripts.kb_manifest import bootstrap; from pathlib import Path; print(bootstrap(Path('data/kb_seed'), reviewed_by='bootstrap-legacy-2026-07-02'), 'documentos registrados')"
```
Expected: N ≥ 8 documentos registrados (todos los .txt actuales).

- [ ] **Step 2: Primer indexado con el embedder nuevo** (migra los IDs al esquema content-addressed)

```bash
python -m scripts.kb_embed
```
Expected: chunks nuevos añadidos y los IDs antiguos (`*_chunkNNN` sin hash) podados; manifest → indexed. Coste: re-embeddings del corpus completo (una vez).

- [ ] **Step 3: Generar el primer lote (2 temas) con el investigador**

```bash
python -m agents.ingest.run_researcher --topic po_operacion_ree
python -m agents.ingest.run_researcher --topic acuerdos_intercambio_fr_pt_ma
```
Expected: 2 `.txt` en draft + entradas en manifest. Revisar el log: si el agente
usó fuentes web, deben aparecer en la sección FUENTES.

- [ ] **Step 4: Validar**

```bash
python -m scripts.kb_validate
```
Expected: exit 0 y ambos → validated. Si falla por `[SIN VERIFICAR]` o capacidad
fuera de allowlist: corregir el documento (o regenerar el tema) — NO relajar el
validador para que pase.

- [ ] **Step 5: GATE HUMANO — el usuario revisa y aprueba**

Presentar los `.txt` al usuario. Tras su OK, por cada documento aprobado:

```bash
python -c "
from scripts import kb_manifest as M
m = M.load_manifest()
for rel in ('regulations/po_operacion_ree.txt', 'procedures/acuerdos_intercambio_fr_pt_ma.txt'):
    M.update_entry(m, rel, status='approved', reviewed_by='Cristian Valverde')
M.save_manifest(m)
print('aprobados')"
```

**PARAR aquí si el usuario pide cambios** — regenerar/editar y volver al Step 4.

- [ ] **Step 6: Indexar los aprobados + idempotencia**

```bash
python -m scripts.kb_embed
python -m scripts.kb_embed
```
Expected: primera pasada añade los chunks nuevos; segunda pasada `+0 chunks, -0 podados` (criterio de aceptación de idempotencia).

- [ ] **Step 7: Smoke de recuperación (integration) + suite offline**

```bash
python -m pytest tests/mcp_servers/test_kb_interconnexiones.py -q -m integration
python -m pytest -q
```
Expected: integration PASS (la recuperación sigue funcionando con el corpus ampliado); suite offline verde.

- [ ] **Step 8: Commit final**

```bash
git add data/kb_seed/
git commit -m "feat(kb): primer lote del pipeline de ingesta (bootstrap + 2 temas nuevos indexados)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Cierre

- [ ] `python -m pytest -q` verde (unit nuevos incluidos).
- [ ] Idempotencia demostrada (Task 5 Step 6, doble ejecución).
- [ ] `grep -rn "gemini-" --include="*.py" agents/ scripts/` → sin nombres de modelo hardcodeados (solo `get_model`).
- [ ] Actualizar `.superpowers/sdd/progress.md` (Plan 6), la memoria `efficacy-ensemble-work` (pendiente RAG resuelto) y `ARCHITECTURE.md` §4 (Documents for RAG: pipeline de ingesta) + §13 (árbol: `agents/ingest/`, `scripts/kb_*`).
- [ ] Los temas restantes de `topics.yaml` (4 pending) quedan para tandas posteriores — el pipeline ya es repetible.

## Orden y dependencias

```
1 (manifest) ──> 2 (validate) ──┐
             └─> 3 (embed) ─────┼──> 5 (ejecución LIVE + gate humano)
4 (researcher + topics) ────────┘
```
Tasks 1-4 son [FRÍO] (offline, suite verde). Task 5 es [LIVE]: Gemini API,
ChromaDB real y aprobación del usuario en el Step 5.
