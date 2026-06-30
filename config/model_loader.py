"""
config/model_loader.py — única fuente de verdad para nombres de modelos Gemini.

NUNCA hardcodear nombres de modelos en el código. Importar siempre desde aquí:

    from config.model_loader import get_model
    model = get_model("fast_agent")   # -> "gemini-3.5-flash"
    model = get_model("smart_agent")  # -> "gemini-3.1-pro"
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml

ModelRole = Literal["fast_agent", "smart_agent", "embedding", "fallback_fast"]

_REGISTRY_PATH = Path(__file__).parent / "model_registry.yaml"


def _load_registry() -> dict:
    with open(_REGISTRY_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_model(role: ModelRole) -> str:
    """Devuelve el nombre de modelo Gemini para el rol dado.

    Lee el registry en cada llamada para reflejar actualizaciones del job semanal
    sin necesidad de reiniciar el proceso.

    Raises:
        KeyError: si el rol no existe en el registry.
        FileNotFoundError: si el registry YAML no existe.
    """
    registry = _load_registry()
    models: dict[str, str] = registry.get("models", {})
    if role not in models:
        available = list(models.keys())
        raise KeyError(
            f"Rol '{role}' no encontrado en model_registry.yaml. "
            f"Roles disponibles: {available}"
        )
    return models[role]


def list_roles() -> dict[str, str]:
    """Devuelve todos los roles y sus modelos asignados actualmente."""
    return dict(_load_registry().get("models", {}))


def get_registry_meta() -> dict:
    """Devuelve metadatos del registry (fecha de actualización, fuente, etc.)."""
    return dict(_load_registry().get("registry_meta", {}))
