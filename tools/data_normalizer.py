"""
tools/data_normalizer.py — Function Tool: normaliza raw dict a GridSnapshot validado.
"""

from __future__ import annotations

from datetime import datetime, timezone

from schemas.models import GridSnapshot


def data_normalizer(raw: dict) -> dict:
    """Convierte unidades a SI, timestamps a UTC, valida schema con Pydantic.

    Args:
        raw: Dict devuelto por ree_poller() u otra fuente de datos de red

    Returns:
        GridSnapshot serializado como dict (JSON-compatible para ADK)
    """
    # Normalizar timestamp a UTC
    ts_raw = raw.get("timestamp") or raw.get("source_timestamp")
    if ts_raw:
        ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
    else:
        ts = datetime.now(timezone.utc)

    # Generación: asegurar valores ≥ 0
    gen_raw = raw.get("generation_by_tech", {})
    generation = {k: max(float(v or 0), 0.0) for k, v in gen_raw.items()}

    # Intercambios: valores positivos = exportación
    inter_raw = raw.get("interchange_flows", {})
    interchange = {k: float(v or 0) for k, v in inter_raw.items()}

    # Demanda: proteger contra valores nulos o negativos
    demand = max(float(raw.get("demand_mw") or 0), 0.01)
    reserve = max(float(raw.get("reserve_mw") or 0), 0.0)

    snapshot = GridSnapshot(
        timestamp=ts,
        region=str(raw.get("region", "peninsular")),
        demand_mw=demand,
        generation_by_tech=generation,
        interchange_flows=interchange,
        reserve_mw=reserve,
        programmed_generation_mw=raw.get("programmed_generation_mw"),
        source_timestamp=ts,
    )
    return snapshot.model_dump(mode="json")
