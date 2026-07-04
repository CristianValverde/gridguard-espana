"""
tools/risk_scorer.py — Function Tool: combina score ML con métricas actuales.

Fórmula (ARCHITECTURE.md §5 Capa 1):
    risk = 0.5 * ml_score + 0.3 * balance_pressure + 0.2 * renewables_risk
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from schemas.models import RiskScore


def risk_scorer(
    snapshot: dict,
    ml_score: float,
    baseline: dict = {},
) -> dict:
    """Calcula el RiskScore combinando el score ML con los proxies actuales.

    Args:
        snapshot:  GridSnapshot como dict
        ml_score:  Anomaly score del modelo ML [0.0–1.0]
        baseline:  Baseline como dict (opcional, mejora la normalización; vacío = sin baseline)

    Returns:
        RiskScore como dict

    Nota: firma sin Optional/None deliberada — ver stability_proxy_calculator
    (mismo motivo: anyOf schema rompe function-calling de Gemini en algunos turnos).
    """
    from tools.stability_proxy_calculator import stability_proxy_calculator

    proxies = stability_proxy_calculator(snapshot, baseline)

    balance_pressure = float(proxies.get("composite_pressure", 0))
    renewables_pct = float(proxies.get("renewables_pct", 0))
    # Riesgo renovables: alta penetración + baja demanda → mayor riesgo
    demand = float(snapshot.get("demand_mw") or 1)
    renewables_risk = min(renewables_pct / 80, 1.0)  # >80% = riesgo máximo

    # Fórmula ponderada definida en ARCHITECTURE.md
    raw_score = (
        0.5 * float(ml_score)
        + 0.3 * balance_pressure
        + 0.2 * renewables_risk
    )
    value = int(min(raw_score * 100, 100))

    # Nivel según umbrales de P(blackout) (correlación empírica con ml_score)
    if value < 25:
        level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = "LOW"
    elif value < 50:
        level = "MEDIUM"
    elif value < 75:
        level = "HIGH"
    else:
        level = "CRITICAL"

    score = RiskScore(
        value=value,
        level=level,
        blackout_probability=round(float(ml_score), 4),
        components={
            "ml_score": round(float(ml_score), 4),
            "balance_pressure": round(balance_pressure, 4),
            "renewables_risk": round(renewables_risk, 4),
        },
        timestamp=datetime.now(timezone.utc),
    )
    return score.model_dump(mode="json")
