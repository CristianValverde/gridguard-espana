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
from agents.instruction_utils import as_static_instruction, load_prompt

REPORTS_DIR = Path(os.environ.get("REPORTS_PATH", "reports"))

_INSTRUCTION = load_prompt("report_agent")


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

report_agent = Agent(
    model=get_model("smart_agent"),
    name="report_agent",
    instruction=as_static_instruction(_INSTRUCTION),
    tools=[
        write_report,
        _memory_toolset,
        _ml_toolset,
        _kb_toolset,
    ],
    generate_content_config=genai_types.GenerateContentConfig(temperature=0.4),
)
