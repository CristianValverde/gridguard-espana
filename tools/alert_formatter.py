"""
tools/alert_formatter.py — Function Tool: construye alerta estructurada.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from schemas.models import Alert


_SCOPE_MAP: dict[str, Literal["nacional", "peninsular", "zonal", "local"]] = {
    "nacional": "nacional",
    "peninsular": "peninsular",
    "zonal": "zonal",
    "local": "local",
}

_RECOMMENDED_ACTIONS: dict[str, dict[str, str]] = {
    "LOW": {
        "nacional": "Monitorización pasiva. Sin acción operacional requerida.",
        "peninsular": "Monitorización pasiva.",
        "zonal": "Monitorización de zona.",
        "local": "Monitorización local.",
    },
    "MEDIUM": {
        "nacional": "Activar reserva terciaria preventiva. Alertar a salas de control.",
        "peninsular": "Revisar programación de generación peninsular.",
        "zonal": "Revisar balance zonal. Considerar redespacho local.",
        "local": "Contactar con el operador de zona.",
    },
    "HIGH": {
        "nacional": (
            "ALERTA WARNING: Activar reserva secundaria y terciaria. "
            "Incrementar interconexiones con Francia y Portugal. "
            "Preparar activación de respuesta de demanda."
        ),
        "peninsular": (
            "Activar reservas de regulación. "
            "Revisar intercambios internacionales."
        ),
        "zonal": "Redespacho urgente en zona afectada. Activar recursos locales.",
        "local": "Desconexión preventiva de cargas no críticas.",
    },
    "CRITICAL": {
        "nacional": (
            "ALERTA CRÍTICA NACIONAL: Activar Plan de Defensa del Sistema REE. "
            "Maximizar importaciones EU. Activar respuesta de demanda de emergencia. "
            "Preparar deslastrado de carga controlado si P(blackout) > 0.7."
        ),
        "peninsular": (
            "Activar todos los recursos de reserva disponibles. "
            "Coordinar con ENTSO-E para importación de emergencia."
        ),
        "zonal": "Deslastrado de carga controlado en zona. Aislar si es necesario.",
        "local": "Desconexión de cargas. Preparar isla de red.",
    },
}


def alert_formatter(
    risk: dict,
    anomalies: list[dict],
    scope: str = "peninsular",
) -> dict:
    """Construye una alerta estructurada a partir del RiskScore y anomalías detectadas.

    Args:
        risk:      RiskScore como dict
        anomalies: Lista de Anomaly dicts (puede estar vacía)
        scope:     Ámbito geográfico ('nacional'|'peninsular'|'zonal'|'local')

    Returns:
        Alert como dict
    """
    severity = str(risk.get("level", "LOW"))
    scope_norm = _SCOPE_MAP.get(scope.lower(), "peninsular")
    blackout_prob = float(risk.get("blackout_probability", 0))
    risk_value = int(risk.get("value", 0))

    # Resumen ejecutivo
    summary = (
        f"[{severity}] Risk Score {risk_value}/100 — "
        f"P(blackout_1h)={blackout_prob:.1%} — Ámbito: {scope_norm}"
    )

    # Detalles: top anomalías detectadas
    if anomalies:
        anomaly_lines = []
        for a in anomalies[:3]:  # máximo 3 anomalías en el detalle
            feature = a.get("feature", "desconocida")
            value = a.get("value", 0)
            desc = a.get("description", "")
            sigma = a.get("deviation_sigma", 0)
            anomaly_lines.append(
                f"  • {feature}: {value:.1f} ({sigma:+.1f}σ) — {desc}"
            )
        details = "Anomalías detectadas:\n" + "\n".join(anomaly_lines)
    else:
        components = risk.get("components", {})
        details = (
            f"Componentes del riesgo: "
            f"ML={components.get('ml_score', 0):.2f}, "
            f"Balance={components.get('balance_pressure', 0):.2f}, "
            f"Renovables={components.get('renewables_risk', 0):.2f}"
        )

    recommended = _RECOMMENDED_ACTIONS.get(severity, {}).get(scope_norm, "Consultar con operador.")

    alert = Alert(
        severity=severity,
        scope=scope_norm,
        timestamp=datetime.now(timezone.utc),
        summary=summary,
        details=details,
        recommended_action=recommended,
    )
    return alert.model_dump(mode="json")
