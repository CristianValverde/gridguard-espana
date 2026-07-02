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
