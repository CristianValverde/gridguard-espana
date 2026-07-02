"""
scripts/check_gemini_models.py — Lista modelos Gemini disponibles y actualiza registry.

Uso:
  $env:GEMINI_API_KEY="tu_clave"
  python scripts/check_gemini_models.py

El script:
  1. Lista todos los modelos disponibles con tu API key.
  2. Selecciona automáticamente los mejores para fast_agent, smart_agent y embedding.
  3. Actualiza config/model_registry.yaml con los nombres reales.
  4. Imprime un resumen legible.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")


def _gemini_key() -> str:
    k = os.environ.get("GEMINI_API_KEY")
    if not k:
        sys.exit("ERROR: GEMINI_API_KEY no definida. Ejecuta: $env:GEMINI_API_KEY='tu_clave'")
    return k


def list_available_models(api_key: str) -> list[dict]:
    """Lista modelos disponibles ordenados por familia (google.genai SDK)."""
    try:
        from google import genai
    except ImportError:
        sys.exit("ERROR: pip install google-genai")

    client = genai.Client(api_key=api_key)
    results = []
    for m in client.models.list():
        name = m.name.replace("models/", "")
        supported = list(getattr(m, "supported_actions", []) or [])
        results.append({
            "name": name,
            "display": getattr(m, "display_name", name),
            "methods": supported,
        })
    return results


def _pick_best(models: list[dict], keywords: list[str],
               method: str = "generateContent") -> str | None:
    """Elige el modelo con mayor relevancia para las keywords dadas."""
    candidates = [m for m in models if method in m.get("methods", [])]

    # Ordenar por preferencia de keywords (primero = más preferido)
    for kw in keywords:
        for c in candidates:
            if kw.lower() in c["name"].lower():
                return c["name"]
    return None


def select_roles(models: list[dict]) -> dict[str, str]:
    """Asigna los 4 roles del registry a los mejores modelos disponibles."""

    # fast_agent: preferir flash más reciente para baja latencia
    fast_prefs = ["2.5-flash", "2.0-flash", "1.5-flash", "flash"]
    fast = _pick_best(models, fast_prefs)

    # smart_agent: preferir pro más reciente para razonamiento complejo
    smart_prefs = ["2.5-pro", "2.0-pro", "1.5-pro", "pro"]
    smart = _pick_best(models, smart_prefs)

    # embedding: preferir modelos de embedding
    emb_prefs = ["embedding-exp", "embedding-2", "embedding"]
    emb = _pick_best(models, emb_prefs, method="embedContent")
    if not emb:
        # fallback: text-embedding-004 (estable, ampliamente disponible)
        emb = "text-embedding-004"

    # fallback_fast: segundo flash disponible (lite o anterior)
    fallback_prefs = ["flash-lite", "flash-8b", "1.5-flash"]
    fallback = _pick_best(models, fallback_prefs) or fast

    return {
        "fast_agent": fast or "gemini-2.0-flash",
        "smart_agent": smart or "gemini-2.0-pro-exp",
        "embedding": emb,
        "fallback_fast": fallback or "gemini-2.0-flash",
    }


def update_registry(roles: dict[str, str]) -> None:
    registry_path = ROOT / "config" / "model_registry.yaml"
    current = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}

    current.setdefault("models", {})
    current["models"].update(roles)
    # Preservar la sección de parámetros si existe
    registry_path.write_text(
        yaml.dump(current, allow_unicode=True, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    print(f"\nRegistry actualizado: {registry_path}")


def main() -> None:
    api_key = _gemini_key()
    print("Conectando con Gemini API…")

    all_models = list_available_models(api_key)

    print(f"\n{'-'*60}")
    print(f"{'MODELOS DISPONIBLES':^60}")
    print(f"{'-'*60}")
    for m in sorted(all_models, key=lambda x: x["name"]):
        methods_str = ", ".join(m["methods"][:2])
        print(f"  {m['name']:<45} [{methods_str}]")

    roles = select_roles(all_models)

    print(f"\n{'-'*60}")
    print(f"{'ASIGNACIÓN AUTOMÁTICA DE ROLES':^60}")
    print(f"{'-'*60}")
    for role, model in roles.items():
        print(f"  {role:<20} → {model}")

    update_registry(roles)

    # Validación final: importar el loader y probar
    from config.model_loader import get_model
    for role in roles:
        resolved = get_model(role)
        print(f"  model_loader.get_model('{role}') = {resolved}  ✓")

    print("\nListo. Ejecuta 'python -m scripts.kb_embed' para poblar el RAG.")


if __name__ == "__main__":
    main()
