# Evals de GridGuard — por agente y por eje

Cada agente se evalúa en tres ejes: **trigger** (¿dispara las tools/skills cuando debe y solo cuando debe?), **output** (¿el resultado es correcto?) y **trayectoria** (¿el orden fue el correcto? shift-left Python→ML→LLM).

Dos niveles:
- **offline** (default, determinista, sin APIs): `python -m pytest eval -q`
- **live/integration** (LLM real + APIs; requiere claves): añadir `-m integration`

## Comandos por agente

| Agente | Offline | Live |
|---|---|---|
| orchestrator | `python -m pytest eval/agents/orchestrator -q` | `python -m pytest eval/agents/orchestrator -m integration` |
| anomaly_detector | `python -m pytest eval/agents/anomaly_detector -q` | `python -m pytest eval/agents/anomaly_detector -m integration` |
| renewables_forecast | `python -m pytest eval/agents/renewables_forecast -q` | `python -m pytest eval/agents/renewables_forecast -m integration` |
| regulatory_compliance | `python -m pytest eval/agents/regulatory_compliance -q` | `python -m pytest eval/agents/regulatory_compliance -m integration` |
| report_agent | `python -m pytest eval/agents/report_agent -q` | `python -m pytest eval/agents/report_agent -m integration` |
| cross_border | `python -m pytest eval/agents/cross_border -q` | `python -m pytest eval/agents/cross_border -m integration` |

**También por marker:** Para usar markers (pytest.ini define `agent_report`, `agent_orchestrator`, etc.), **el comando correcto incluye `not integration`:**

```bash
python -m pytest -m "agent_report and not integration" -q
```

Nota técnica: En CLI, `-m agent_report` sin restricción de integration ejecutaría los tests `@pytest.mark.integration` asociados al marker, **sobrescribiendo** el `addopts = -m "not integration"` del pytest.ini (CLI gana sobre config). Por eso es obligatorio incluir `and not integration` al usar markers. Por path, en cambio (e.g., `pytest eval/agents/report_agent`), el `addopts` se respeta y los tests integration se deseleccionan automáticamente.

## Advertencia: Tests live/integration

Los tests con `-m integration` requieren **claves de API reales** (GOOGLE_API_KEY, tokens MCP) y consumen cuota LLM. A la fecha (2026-07-02) **no se han ejecutado contra LLM real en CI/CD** — su validación está pendiente. Úsalos solo en entorno local con tokens configurados.

## Flujo prompt→eval

Los prompts del sistema viven en `prompts/<agente>.md`. Tras cambiar un prompt:

1. Edita `prompts/<agente>.md`
2. Corre el eval offline: `python -m pytest eval/agents/<agente> -q`
3. Verifica que los tests pasen
4. Si todos pasan, corre el eval live (si aplica): `python -m pytest eval/agents/<agente> -m integration`
5. Compara resultados antes de commitear

El sistema evalúa tanto **qué tools dispara** (trigger) como **el orden** en que las llamó (trayectoria) y **si el resultado es correcto** (output).

## Extensión futura

Framework nativo de eval de ADK (`.evalset.json` + `agents-cli eval`, LLM-as-judge). No implementado: la suite actual es offline/determinista por diseño (shift-left Python→ML→LLM). Ver spec 2026-07-02.

## Estado de la validación live (primera ejecución: 2026-07-02/03)

Dos corridas completas contra LLM real. Hallazgos corregidos: whitelist del
orchestrator bloqueaba las skills (nombres CamelCase vs runtime snake_case),
timeout MCP de 5s insuficiente (ahora 60s), test de Markov con import obsoleto.

Estado por test tras los fixes: 10 de 12 han pasado en al menos una corrida.
`orchestrator/test_critical` y `renewables/test_load_forecast` nunca han pasado:
la causa dominante es la tormenta de subprocesos MCP (cada test del orchestrator
arranca ~15-20 procesos stdio simultáneos que importan torch a la vez; el import
en frío son ~6s sin carga, pero bajo contención las llamadas superan 60s y el
orchestrator activa su fallback "ML caído"). Los tests de subagentes individuales
(anomaly, cross_border) muestran además no-determinismo LLM normal (~1 flip por
corrida). Trabajo futuro: transporte MCP compartido o sesiones reutilizadas para
eliminar la tormenta de arranque antes de endurecer estos evals.
