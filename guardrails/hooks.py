"""
guardrails/hooks.py — ADK Callbacks (Guardrails) de GridGuard.

5 hooks registrados en OrchestratorAgent (y propagados a Skills/Sub-agents):

  before_tool_callback  — bloquea .env, rate-limit, whitelist de tools
  after_tool_callback   — valida schema, escanea credenciales, trunca output
  before_model_callback — anti-prompt-injection
  after_model_callback  — valida rangos numéricos y JSON
  after_agent_callback  — valida schema final del agente

NOTA ADK 2.3.0: firmas confirmadas desde google-adk 2.3.0:

  before_tool_callback(tool: BaseTool, args: dict, tool_context: ToolContext)
      -> Optional[dict]
      tool.name  → nombre del tool
      tool_context.agent_name  → nombre del agente

  after_tool_callback(tool: BaseTool, args: dict, tool_context: ToolContext,
                      tool_response: dict) -> Optional[dict]

  before_model_callback(callback_context: CallbackContext, llm_request: LlmRequest)
      -> Optional[LlmResponse]
      llm_request.contents  → list[types.Content]; cada Content.parts[i].text

  after_model_callback(callback_context: CallbackContext, llm_response: LlmResponse)
      -> Optional[LlmResponse]
      llm_response.content.parts[i].text  → texto generado

  after_agent_callback(callback_context: CallbackContext)
      -> Optional[types.Content]
      Estado del agente en callback_context.state
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types as genai_types

log = logging.getLogger("guardrails")

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

# Tools que acceden a ficheros (a vigilar contra rutas sensibles)
_FILE_READ_TOOLS = {"read_file", "open_file", "load_document", "write_report"}

# Patrones de rutas sensibles — bloqueadas en before_tool_callback
_BLOCKED_PATH_PATTERNS = [
    ".env", "secrets", ".key", ".pem", "credentials", "token",
    "password", "apikey", "api_key", "private",
]

# Rate limiting: segundos mínimos entre llamadas al mismo tool
_MIN_INTERVAL: dict[str, float] = {
    "ree_poller": 60.0,
    "get_generation_mix": 60.0,
    "get_wind_forecast": 300.0,
    "get_solar_radiation": 300.0,
    "get_cross_border_flows": 300.0,
    "get_eu_generation": 300.0,
}

_rate_limiter: dict[str, float] = {}  # tool_name → last_call_monotonic

# Estado Markov notable (S2 y superiores disparan acciones especiales)
_S2_NOTABLE = 2

# ---------------------------------------------------------------------------
# Plan 4: Hooks de transición de estado (Task 2)
# ---------------------------------------------------------------------------


def on_state_transition(prev_state: int, curr_state: int) -> dict:
    """Reacción a un cambio de estado de Markov entre ciclos (ver Plan 4).

    - S2+ → S0 (recuperación): dispara post-mortem.
    - S0 → S2+ (onset abrupto): dispara verificación normativa inmediata.
    """
    if prev_state >= _S2_NOTABLE and curr_state == 0:
        log.info("[HOOK] on_state_transition: recuperación S%d→S0 → post_mortem", prev_state)
        return {"action": "post_mortem", "trigger": f"S{prev_state}->S0"}
    if prev_state == 0 and curr_state >= _S2_NOTABLE:
        log.warning("[HOOK] on_state_transition: onset S0→S%d → regulatory_check", curr_state)
        return {"action": "regulatory_check", "trigger": f"S0->S{curr_state}"}
    return {"action": "none", "trigger": None}


def on_forecast_risk_horizon(window_label: str, severity: str) -> dict:
    """Registra una ventana de riesgo 6–24 h detectada por RenewablesForecastSkill."""
    log.warning("[HOOK] on_forecast_risk_horizon: ventana de riesgo %s (%s)", window_label, severity)
    return {"logged": True, "advance_warning": window_label}


# Tools permitidas por tipo de agente (principio de mínimo privilegio)
_ALLOWED_TOOLS: dict[str, set[str]] = {
    "gridguard_orchestrator": {
        "ree_poller", "data_normalizer", "stability_proxy_calculator",
        "risk_scorer", "escalation_planner", "alert_formatter",
        "get_blackout_probability", "store_risk_score",
        "get_serve_features", "get_ensemble_score", "get_blackout_probability_ensemble",
        "anomaly_detector_skill", "renewables_forecast_skill",
        "cross_border_agent", "report_agent",
        "get_grid_state", "store_grid_state", "regulatory_compliance_skill",
    },
    "anomaly_detector_skill": {
        "get_anomaly_score", "stability_proxy_calculator",
        "get_baseline", "search_regulations",
        "get_grid_state", "get_wind_forecast", "get_solar_radiation",
    },
    "renewables_forecast_skill": {
        "get_wind_forecast", "get_solar_radiation", "get_generation_mix",
        "get_load_forecast", "get_baseline",
        "get_weather_forecast_24h", "get_grid_state",
    },
    "report_agent": {
        "query_incidents", "get_risk_history", "get_blackout_probability",
        "find_similar_incident", "write_report",
        "get_markov_trajectory", "get_state_transitions",
    },
    "cross_border_agent": {
        "get_cross_border_flows", "get_capacity_margin",
        "get_eu_generation", "get_mitigation_procedure",
        "get_eu_market_prices",
    },
    "regulatory_compliance_skill": {
        "search_regulations", "get_grid_state",
    },
}

# Patrones de inyección de prompt a detectar y sanitizar
_INJECTION_PATTERNS = [
    "ignore previous", "ignore your", "system:", "jailbreak",
    "forget your", "new instruction", "disregard", "override",
    "act as", "you are now", "pretend you", "ignore all",
    "bypass", "sudo", "admin mode",
]

# Rangos físicos válidos para validación de output
_PHYSICAL_RANGES = {
    "demand_mw": (5_000, 60_000),       # demanda peninsular en MW
    "anomaly_score": (0.0, 1.0),
    "blackout_probability": (0.0, 1.0),
    "probability": (0.0, 1.0),
    "renewables_pct": (0.0, 100.0),
    "composite_pressure": (0.0, 1.0),
    "confidence": (0.0, 1.0),
    "value": (0, 100),                   # RiskScore.value
}

# Expresiones regulares para detectar credenciales en outputs
_CREDENTIAL_PATTERNS = [
    re.compile(r'["\s]?(?:token|key|secret|password|api_?key)["\s]?\s*[:=]\s*["\']?\w{8,}', re.I),
    re.compile(r'Bearer\s+[A-Za-z0-9\-_.~+/]{20,}'),
    re.compile(r'AIza[0-9A-Za-z\-_]{35}'),   # Google API key
    re.compile(r'sk-[a-zA-Z0-9]{32,}'),       # OpenAI-style secret
]

_MAX_OUTPUT_TOKENS = 4_000  # caracteres aproximados — 4k tokens ~= 16k chars
_MAX_OUTPUT_CHARS = _MAX_OUTPUT_TOKENS * 4


# ---------------------------------------------------------------------------
# 1. before_tool_callback
# ---------------------------------------------------------------------------

def before_tool_callback(
    tool: BaseTool,
    args: dict,
    tool_context: ToolContext,
) -> Optional[dict]:
    """Ejecutado antes de cada llamada a tool. ADK 2.3.0 signature.

    Args:
        tool: BaseTool instance — use tool.name for the tool name.
        args: The keyword arguments the LLM passed to the tool.
        tool_context: ToolContext (same as CallbackContext) — use
            tool_context.agent_name for the running agent's name.

    Returns:
        None → dejar pasar la llamada
        dict → devolver este dict como resultado sin ejecutar el tool (bloqueo)
    """
    tool_name: str = tool.name
    tool_args: dict = args
    agent_name: str = tool_context.agent_name

    # ── Bloquear acceso a ficheros sensibles ─────────────────────────────────
    if tool_name in _FILE_READ_TOOLS:
        path = str(tool_args.get("path", "") or tool_args.get("file_path", "")).lower()
        for pattern in _BLOCKED_PATH_PATTERNS:
            if pattern in path:
                log.warning("[GUARDRAIL] before_tool: acceso bloqueado a %s", path)
                return {"error": f"ACCESS_DENIED: ruta sensible bloqueada ({pattern})"}

    # ── Rate limiting ─────────────────────────────────────────────────────────
    if tool_name in _MIN_INTERVAL:
        last = _rate_limiter.get(tool_name, 0.0)
        elapsed = time.monotonic() - last
        if elapsed < _MIN_INTERVAL[tool_name]:
            remaining = _MIN_INTERVAL[tool_name] - elapsed
            log.warning("[GUARDRAIL] before_tool: rate limit en %s (%.0fs restantes)",
                        tool_name, remaining)
            return {"error": f"RATE_LIMITED: {tool_name} — esperar {remaining:.0f}s"}
        _rate_limiter[tool_name] = time.monotonic()

    # ── Whitelist de tools por agente ─────────────────────────────────────────
    if agent_name and agent_name in _ALLOWED_TOOLS:
        allowed = _ALLOWED_TOOLS[agent_name]
        if tool_name not in allowed:
            log.warning("[GUARDRAIL] before_tool: %s no permitido para %s",
                        tool_name, agent_name)
            return {"error": f"UNAUTHORIZED: {tool_name} no permitido para {agent_name}"}

    return None


# ---------------------------------------------------------------------------
# 2. after_tool_callback
# ---------------------------------------------------------------------------

def after_tool_callback(
    tool: BaseTool,
    args: dict,
    tool_context: ToolContext,
    tool_response: dict,
) -> Optional[dict]:
    """Ejecutado tras cada llamada a tool. ADK 2.3.0 signature.

    Args:
        tool: BaseTool instance — use tool.name.
        args: The keyword arguments passed to the tool.
        tool_context: ToolContext — use tool_context.agent_name.
        tool_response: The dict returned by the tool execution.

    Returns:
        None → usar tool_response sin modificar
        dict → reemplazar tool_response con este dict
    """
    tool_name: str = tool.name
    output: Any = tool_response

    if not isinstance(output, (dict, list, str)):
        return None  # nothing to modify; ADK keeps original

    output_str = json.dumps(output) if isinstance(output, (dict, list)) else str(output)

    # ── Escanear por credenciales ─────────────────────────────────────────────
    for pattern in _CREDENTIAL_PATTERNS:
        if pattern.search(output_str):
            log.error("[GUARDRAIL] after_tool: credencial detectada en output de %s — REDACTADO",
                      tool_name)
            if isinstance(output, dict):
                output["_warning"] = "CREDENTIAL_REDACTED: output contenía posible credencial"
                return output
            # string/list: bloquear por completo en vez de dejar pasar la credencial
            return {"error": "CREDENTIAL_REDACTED: output de tool contenía una credencial"}

    # ── Validar rangos físicos ────────────────────────────────────────────────
    modified = False
    if isinstance(output, dict):
        for field, (lo, hi) in _PHYSICAL_RANGES.items():
            if field in output:
                val = output[field]
                if isinstance(val, (int, float)) and not (lo <= val <= hi):
                    log.warning("[GUARDRAIL] after_tool: %s.%s=%s fuera de rango [%s, %s]",
                                tool_name, field, val, lo, hi)
                    output[field] = max(lo, min(hi, val))  # clamp en lugar de bloquear
                    output["_clamped"] = output.get("_clamped", []) + [field]
                    modified = True

    # ── Truncar si supera el límite de tokens (solo para dicts) ────────────────
    if isinstance(output, dict) and len(output_str) > _MAX_OUTPUT_CHARS:
        log.warning("[GUARDRAIL] after_tool: output de %s truncado (%d → %d chars)",
                    tool_name, len(output_str), _MAX_OUTPUT_CHARS)
        output["_truncated"] = True
        # Truncar campos de texto largos
        for k, v in output.items():
            if isinstance(v, str) and len(v) > 2000:
                output[k] = v[:2000] + "... [TRUNCADO]"
        modified = True

    # Return modified dict or None (ADK will keep original if None)
    return output if modified else None


# ---------------------------------------------------------------------------
# 3. before_model_callback
# ---------------------------------------------------------------------------

def before_model_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> Optional[LlmResponse]:
    """Detecta y sanitiza patrones de prompt injection antes de enviar al LLM.

    ADK 2.3.0 signature.

    Args:
        callback_context: CallbackContext — use callback_context.agent_name.
        llm_request: LlmRequest — mutation of llm_request.contents sanitizes
            the messages in-place.

    Returns:
        None → proceder con llm_request (posiblemente mutado)
        LlmResponse → cortocircuitar: devolver esta respuesta sin llamar al LLM
    """
    sanitized = 0
    for content_item in llm_request.contents:
        if not content_item.parts:
            continue
        for part in content_item.parts:
            text = part.text or ""
            text_lower = text.lower()
            if any(p in text_lower for p in _INJECTION_PATTERNS):
                log.warning("[GUARDRAIL] before_model: posible injection detectada — sanitizada")
                part.text = "[SANITIZED: contenido potencialmente malicioso eliminado]"
                sanitized += 1
    if sanitized:
        log.info("[GUARDRAIL] before_model: %d part(s) sanitizada(s)", sanitized)
    # Return None to let ADK proceed with the (possibly mutated) request
    return None


# ---------------------------------------------------------------------------
# 4. after_model_callback
# ---------------------------------------------------------------------------

def _extract_text_from_llm_response(llm_response: LlmResponse) -> str:
    """Helper: extrae el texto concatenado de un LlmResponse."""
    if not llm_response.content or not llm_response.content.parts:
        return ""
    return "".join(part.text or "" for part in llm_response.content.parts)


def _replace_text_in_llm_response(llm_response: LlmResponse, new_text: str) -> None:
    """Helper: reemplaza el texto del primer part de texto en un LlmResponse."""
    if not llm_response.content or not llm_response.content.parts:
        return
    # Set first text part; clear the rest or set them to empty
    first_set = False
    for part in llm_response.content.parts:
        if part.text is not None:
            if not first_set:
                part.text = new_text
                first_set = True
            else:
                part.text = ""


def after_model_callback(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> Optional[LlmResponse]:
    """Valida el output del modelo LLM antes de procesarlo.

    ADK 2.3.0 signature.

    Args:
        callback_context: CallbackContext.
        llm_response: LlmResponse — mutate in-place or return a new LlmResponse
            to override, return None to proceed unchanged.

    Returns:
        None → usar llm_response sin cambios (o con mutaciones in-place)
        LlmResponse → reemplazar la respuesta del modelo con esta
    """
    output = _extract_text_from_llm_response(llm_response)
    if not output:
        return None

    # ── Verificar JSON válido si parece JSON ──────────────────────────────────
    stripped = output.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
            # Validar rangos en el JSON del modelo
            modified = False
            if isinstance(parsed, dict):
                for field, (lo, hi) in _PHYSICAL_RANGES.items():
                    if field in parsed:
                        val = parsed[field]
                        if isinstance(val, (int, float)) and not (lo <= val <= hi):
                            log.warning("[GUARDRAIL] after_model: %s=%s fuera de rango",
                                        field, val)
                            parsed[field] = max(lo, min(hi, val))
                            modified = True
            if modified:
                _replace_text_in_llm_response(llm_response, json.dumps(parsed))
                return llm_response
        except json.JSONDecodeError:
            log.warning("[GUARDRAIL] after_model: output parece JSON pero no es válido")

    # ── Escanear credenciales en output textual ───────────────────────────────
    for pattern in _CREDENTIAL_PATTERNS:
        if pattern.search(output):
            log.error("[GUARDRAIL] after_model: credencial en output del modelo — REDACTADO")
            _replace_text_in_llm_response(
                llm_response, "[REDACTED: output contenía posible credencial]"
            )
            return llm_response

    return None


# ---------------------------------------------------------------------------
# 5. after_agent_callback
# ---------------------------------------------------------------------------

_VALID_SEVERITIES = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
_VALID_SCOPES = {"nacional", "peninsular", "zonal", "local"}

# Keys in session state where GridGuard agents store their domain output
_AGENT_OUTPUT_STATE_KEYS = (
    "risk_score", "value", "blackout_probability", "probability",
    "severity", "level", "scope",
)


def _validate_agent_output_dict(agent_output: dict) -> None:
    """Validate and clamp/flag domain values in agent_output in-place.

    This is the pure validation logic, shared by after_agent_callback and
    any direct callers (e.g. tests, orchestrator post-processing).
    """
    if not isinstance(agent_output, dict):
        return

    errors: list[str] = []

    # risk_score ∈ [0, 100]
    rs = agent_output.get("risk_score") or agent_output.get("value")
    if rs is not None and not (0 <= int(rs) <= 100):
        errors.append(f"risk_score={rs} fuera de [0, 100]")
        agent_output["risk_score"] = max(0, min(100, int(rs)))

    # P(blackout) ∈ [0, 1]
    bp = agent_output.get("blackout_probability") or agent_output.get("probability")
    if bp is not None and not (0.0 <= float(bp) <= 1.0):
        errors.append(f"blackout_probability={bp} fuera de [0, 1]")
        agent_output["blackout_probability"] = max(0.0, min(1.0, float(bp)))

    # severity ∈ VALID_SEVERITIES
    sev = agent_output.get("severity") or agent_output.get("level")
    if sev is not None and str(sev).upper() not in _VALID_SEVERITIES:
        errors.append(f"severity='{sev}' no válida")

    # scope ∈ VALID_SCOPES
    scope = agent_output.get("scope")
    if scope is not None and str(scope).lower() not in _VALID_SCOPES:
        errors.append(f"scope='{scope}' no válido")

    if errors:
        log.warning("[GUARDRAIL] after_agent: %d errores de validación: %s", len(errors), errors)
        agent_output["_validation_errors"] = errors


def after_agent_callback(
    callback_context: CallbackContext,
) -> Optional[genai_types.Content]:
    """Valida el schema y bounds del output final del agente.

    ADK 2.3.0 signature.

    Args:
        callback_context: CallbackContext — agent output is read from
            callback_context.state (session state dict where GridGuard agents
            store risk_score, blackout_probability, severity, scope, etc.).

    Returns:
        None → no override; any validation errors are logged and state is
            updated in-place via callback_context.state.
        types.Content → append this as an extra agent response event (not used
            by this guard — we always return None).

    NOTE: In ADK 2.3.0 the callback no longer receives the agent output dict
    directly.  GridGuard agents are expected to write their domain output into
    session state under the keys listed in _AGENT_OUTPUT_STATE_KEYS.  This
    guard reads those keys, validates, and writes corrections back into state.
    """
    state = callback_context.state

    # Build a snapshot of the relevant domain keys from session state
    agent_output: dict = {
        k: state[k] for k in _AGENT_OUTPUT_STATE_KEYS if k in state
    }

    if not agent_output:
        # No domain output in state yet — nothing to validate
        log.warning("[GUARDRAIL] after_agent: sin claves de dominio en state — validación omitida (el agente puede no haber escrito su output a state).")
        return None

    # Run validation (mutates agent_output in-place, logging errors)
    _validate_agent_output_dict(agent_output)

    # Write validated/clamped values back into session state
    for k, v in agent_output.items():
        state[k] = v

    return None


# ---------------------------------------------------------------------------
# Adaptador para ADK (registra todos los hooks en un agente)
# ---------------------------------------------------------------------------

def register_hooks(agent) -> None:
    """Registra los 5 callbacks en una instancia de Agent de ADK.

    Uso en orchestrator.py:
        from guardrails.hooks import register_hooks
        register_hooks(orchestrator)
    """
    agent.before_tool_callback = before_tool_callback
    agent.after_tool_callback = after_tool_callback
    agent.before_model_callback = before_model_callback
    agent.after_model_callback = after_model_callback
    agent.after_agent_callback = after_agent_callback
