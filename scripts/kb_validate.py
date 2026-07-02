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
