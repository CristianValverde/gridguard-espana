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
