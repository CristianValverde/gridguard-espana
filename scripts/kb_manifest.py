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
