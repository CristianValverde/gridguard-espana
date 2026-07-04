"""
tools/stability_proxy_calculator.py — Function Tool: calcula proxies de estabilidad.

Los datos de frecuencia real (Hz) no están disponibles públicamente vía API.
Este tool calcula 4 proxies estadísticos que en la literatura académica son
los features estándar para predicción de inestabilidad en redes con alta
penetración renovable. Ver ARCHITECTURE.md §5, Capa 1.
"""

from __future__ import annotations

import math

from schemas.models import StabilityProxies


# ---------------------------------------------------------------------------
# Función canónica compartida — fuente única de verdad para composite_pressure
# ---------------------------------------------------------------------------

def compute_composite_pressure_score(
    balance_mw: float,
    reserve_activated_pct: float,
    interchange_delta_mw: float,
    renewables_pct: float,
) -> float:
    """Calcula el índice de presión compuesta [0, 1] con umbrales operacionales absolutos.

    Esta es la fórmula canónica usada en entrenamiento, servicio y validación.
    NO usar la rama sigmoid (baseline) en producción para evitar train/serve skew.

    Args:
        balance_mw:             Diferencia generación - demanda (MW). Positivo = exceso.
        reserve_activated_pct:  Reserva regulación activada (0-100 %).
        interchange_delta_mw:   Variación absoluta en flujo neto de intercambio (MW).
        renewables_pct:         % renovables no gestionables sobre demanda (0-100).

    Returns:
        float en [0, 1]. Valores altos = mayor presión sobre la red.
    """
    balance_score = min(abs(balance_mw) / 3000.0, 1.0)          # >3000 MW = crítico
    reserve_score = reserve_activated_pct / 100.0
    interchange_score = min(interchange_delta_mw / 2000.0, 1.0)
    renewables_score = min(renewables_pct / 80.0, 1.0)           # >80 % = muy alto

    composite = (
        0.35 * balance_score
        + 0.25 * reserve_score
        + 0.20 * interchange_score
        + 0.20 * renewables_score
    )
    return max(0.0, min(composite, 1.0))


def stability_proxy_calculator(snapshot: dict, baseline: dict = {}) -> dict:
    """Calcula los 4 proxies de frecuencia y el índice compuesto.

    Args:
        snapshot: GridSnapshot como dict (salida de data_normalizer)
        baseline: Baseline como dict (salida de get_baseline). Si vacío, sin normalización.

    Nota: firma sin Optional/None deliberada — un tipo `dict | None` genera un
    schema JSON `anyOf` en la function declaration que Gemini function-calling
    a veces falla en formatear (finish_reason=MALFORMED_FUNCTION_CALL,
    verificado en vivo 2026-07-04). `{}` es falsy igual que None para el
    `if baseline:` de abajo, así que el comportamiento no cambia.

    Returns:
        StabilityProxies como dict
    """
    gen = snapshot.get("generation_by_tech", {})
    interchange = snapshot.get("interchange_flows", {})
    demand = float(snapshot.get("demand_mw") or 1)
    reserve = float(snapshot.get("reserve_mw") or 0)

    # 1. Balance oferta/demanda (MW) — proxy más directo de presión de frecuencia
    total_gen = sum(float(v or 0) for v in gen.values())
    balance_mw = total_gen - demand

    # 2. % Reserva de regulación activada
    # Capacidad máxima de reserva secundaria ≈ 1.5% de la demanda (estándar ENTSO-E)
    max_reserve_capacity = demand * 0.015
    reserve_activated_pct = min(
        abs(reserve) / max(max_reserve_capacity, 1) * 100, 100.0
    )

    # 3. Variación brusca en flujos netos de intercambio
    # Aquí calculamos el valor absoluto del flujo neto total
    # (el OrchestratorAgent guarda el valor anterior para calcular la delta real)
    net_interchange = sum(float(v or 0) for v in interchange.values())
    interchange_delta_mw = abs(net_interchange)

    # 4. % renovables no gestionables sobre demanda
    wind = float(gen.get("wind_mw") or 0)
    solar_pv = float(gen.get("solar_pv_mw") or 0)
    solar_thermal = float(gen.get("solar_thermal_mw") or 0)
    renewables_mw = wind + solar_pv + solar_thermal
    renewables_pct = min(renewables_mw / max(demand, 1) * 100, 100.0)

    # 5. Presión compuesta [0–1] — combinación ponderada normalizada
    # Normalización con estadísticas del baseline si disponible
    if baseline:
        b_balance_mean = float(baseline.get("balance_mean_mw") or 0)
        b_balance_std = float(baseline.get("balance_std_mw") or 1)
        b_renew_mean = float(baseline.get("renewables_pct_mean") or 0)
        b_renew_std = float(baseline.get("renewables_pct_std") or 1)

        # Desviación normalizada (valor positivo = más anómalo)
        balance_z = abs(balance_mw - b_balance_mean) / max(b_balance_std, 1)
        renew_z = max(renewables_pct - b_renew_mean, 0) / max(b_renew_std, 1)

        # Normalización sigmoide para acotar a [0, 1]
        def _sigmoid(x: float) -> float:
            return 1 / (1 + math.exp(-0.5 * (x - 2)))

        balance_score = _sigmoid(balance_z)
        reserve_score = reserve_activated_pct / 100
        interchange_score = min(interchange_delta_mw / 2000, 1.0)
        renewables_score = _sigmoid(renew_z)
    else:
        # Sin baseline: usa la fórmula canónica absoluta — RUTA PRINCIPAL para
        # consistencia train/serve. La rama sigmoid (if baseline) NO es la ruta
        # por defecto; evitar en producción salvo que se regenere el dataset
        # con la misma normalización.
        return_early_composite = compute_composite_pressure_score(
            balance_mw, reserve_activated_pct, interchange_delta_mw, renewables_pct
        )
        proxies = StabilityProxies(
            balance_mw=round(balance_mw, 2),
            reserve_activated_pct=round(reserve_activated_pct, 2),
            interchange_delta_mw=round(interchange_delta_mw, 2),
            renewables_pct=round(renewables_pct, 2),
            composite_pressure=round(return_early_composite, 4),
        )
        return proxies.model_dump(mode="json")

    # Rama baseline (sigmoid) — solo se llega aquí cuando baseline is not None
    # Pesos: balance > reserva > intercambio > renovables (según literatura)
    composite_pressure = (
        0.35 * balance_score
        + 0.25 * reserve_score
        + 0.20 * interchange_score
        + 0.20 * renewables_score
    )

    proxies = StabilityProxies(
        balance_mw=round(balance_mw, 2),
        reserve_activated_pct=round(reserve_activated_pct, 2),
        interchange_delta_mw=round(interchange_delta_mw, 2),
        renewables_pct=round(renewables_pct, 2),
        composite_pressure=round(min(composite_pressure, 1.0), 4),
    )
    return proxies.model_dump(mode="json")
