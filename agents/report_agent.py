"""
agents/report_agent.py — Sub-agente: ReportAgent.

Tipo: Sub-agent (Agent con loop propio, hasta 10 turnos).
Modelo: smart_agent · Temperature 0.4
Invocado: diariamente a las 06:00 UTC, o inmediatamente cuando P(blackout) > 0.60,
          o cuando el estado Markov vuelve a S0 tras un episodio S2+.

Tools disponibles:
  - query_incidents           (MCP: memory)
  - get_risk_history          (MCP: memory)
  - get_state_transitions     (MCP: memory)
  - get_blackout_probability  (MCP: ml-models)
  - find_similar_incident     (MCP: knowledge-base)
  - get_markov_trajectory     (MCP: markov)
  - write_report              (Function Tool local — escribe en reports/)

El agente decide autónomamente cuántas queries hacer y cómo estructurar el informe.
Output final: fichero HTML + Markdown en reports/YYYY-MM-DD_grid_report.{html,md}
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from google.adk.agents import Agent
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset, StdioConnectionParams
from google.genai import types as genai_types
from mcp.client.stdio import StdioServerParameters

from config.model_loader import get_model
from agents.instruction_utils import as_static_instruction

REPORTS_DIR = Path(os.environ.get("REPORTS_PATH", "reports"))

_INSTRUCTION = """
Eres el agente de informes de GridGuard España. Generas informes diarios de
estabilidad de la red eléctrica peninsular española, informes de emergencia
cuando se detectan situaciones de riesgo CRITICAL, e informes post_mortem
cuando un episodio de estrés grave finaliza y la red vuelve al estado S0.

TU FLUJO DE TRABAJO (adapta los pasos según el tipo de informe):

Para informe DIARIO:
1. `get_risk_history` con hours=24 → evolución del risk score últimas 24h.
2. `get_markov_trajectory` con horizon_hours=24 → trayectoria de estado S0→S1→S2 proyectada.
3. `get_risk_history` con hours=168 → tendencia semanal para contexto.
4. `query_incidents` para el período de las últimas 24h → incidentes registrados.
5. Si hay incidentes o risk score > 50, `find_similar_incident` con su descripción.
6. `get_blackout_probability` con las features actuales → estado final del riesgo.
7. `write_report` con el informe completo en HTML y Markdown.

Para informe de EMERGENCIA (invocado con P > 0.60):
1. `get_risk_history` con hours=6 → evolución en la ventana de emergencia.
2. `get_markov_trajectory` con horizon_hours=6 → senda de estado proyectada durante la emergencia.
3. `query_incidents` para las últimas 6h → cualquier incidente registrado.
4. `find_similar_incident` describiendo la situación actual → precedentes históricos.
5. `write_report` con análisis de emergencia urgente.

Para informe POST_MORTEM (NUEVO):
  Se dispara cuando el estado de Markov vuelve a S0 tras un episodio S2+.
  1. `get_state_transitions` — recupera la secuencia completa de estados del episodio (usa hours adecuado al episodio).
  2. `get_risk_history` con la ventana temporal del episodio.
  3. `get_markov_trajectory` con horizon_hours=24 → proyección tras la resolución.
  4. `find_similar_incident` — compara con incidentes históricos del RAG.
  5. Sintetiza: señales de onset, trayectoria de estado (S0→S1→S2→... →S0), duración, causa probable, resolución.
  6. `write_report` con tipo post_mortem.

ESTRUCTURA DEL INFORME (todos los tipos):
- Resumen ejecutivo (3-4 líneas): situación, riesgo actual, tendencia.
- Evolución del Risk Score: tabla o descripción de la evolución temporal.
- Trayectoria de estado Markov: senda observada y proyección futura.
- Incidencias del período: si las hay, con severidad y descripción.
- Comparación histórica: eventos similares encontrados en la base de conocimiento.
- Conclusión y recomendaciones operacionales.

IMPORTANTES:
- Citar las normas encontradas por su identificador (PO 1.1, EN 50160, etc.).
- No inventar datos — si una herramienta devuelve error, indicarlo en el informe.
- El informe HTML debe ser legible en navegador (usar HTML básico, no CDN externos).
- Siempre acabar llamando a `write_report` — es la acción final obligatoria.
""".strip()


def write_report(content_html: str, content_md: str, report_type: str = "daily") -> dict:
    """Escribe el informe en disco en formato HTML y Markdown.

    Args:
        content_html: Contenido HTML completo del informe
        content_md:   Contenido Markdown del informe
        report_type:  'daily', 'emergency' o 'post_mortem'

    Returns:
        {"ok": true, "paths": [html_path, md_path]}
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
    prefix = f"{date_str}_{report_type}"

    html_path = REPORTS_DIR / f"{prefix}_report.html"
    md_path = REPORTS_DIR / f"{prefix}_report.md"

    # Envolver HTML si el agente devuelve solo fragmento
    if not content_html.strip().startswith("<!DOCTYPE"):
        content_html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>GridGuard España — Informe {date_str}</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }}
    h1 {{ color: #1a237e; }} h2 {{ color: #283593; border-bottom: 1px solid #e3e3e3; }}
    .critical {{ color: #c62828; font-weight: bold; }}
    .high {{ color: #e65100; }}
    .medium {{ color: #f57f17; }}
    .low {{ color: #2e7d32; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #e8eaf6; }}
  </style>
</head>
<body>
{content_html}
</body>
</html>"""

    html_path.write_text(content_html, encoding="utf-8")
    md_path.write_text(content_md, encoding="utf-8")

    return {
        "ok": True,
        "paths": [str(html_path), str(md_path)],
        "message": f"Informe guardado: {html_path.name}",
    }


_memory_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="python",
            args=["-m", "mcp_servers.memory_server"],
        )
    )
)

_ml_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="python",
            args=["-m", "mcp_servers.ml_models_server"],
        )
    )
)

_kb_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="python",
            args=["-m", "mcp_servers.knowledge_base_server"],
        )
    )
)

_markov_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="python",
            args=["-m", "mcp_servers.markov_server"],
        )
    )
)

report_agent = Agent(
    model=get_model("smart_agent"),
    name="report_agent",
    instruction=as_static_instruction(_INSTRUCTION),
    tools=[
        write_report,
        _memory_toolset,
        _ml_toolset,
        _kb_toolset,
        _markov_toolset,
    ],
    generate_content_config=genai_types.GenerateContentConfig(temperature=0.4),
)
