"""
mcp_servers/weather_server.py — MCP: weather

Datos meteorológicos para zonas de generación renovable.
Primario: AEMET OpenData (token requerido).
Fallback: Open-Meteo (gratuito, sin token) si AEMET falla o no responde.

Credencial: AEMET_API_KEY (variable de entorno).

Herramientas expuestas:
  get_wind_forecast(zone, hours)    -> WindForecast (JSON)
  get_solar_radiation(zone, hours)  -> RadiationForecast (JSON)
  get_aemet_alerts()                -> list[MeteoAlert] (JSON)

Zonas disponibles: norte, sur, levante, interior, galicia, nordeste
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")
from typing import Any

import requests
from mcp.server.fastmcp import FastMCP

from schemas.models import MeteoAlert, RadiationForecast, WindForecast

mcp = FastMCP("gridguard-weather")

AEMET_BASE = "https://opendata.aemet.es/opendata/api"
OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"
CACHE_TTL = 1800  # 30 min — datos meteorológicos cambian más despacio

_cache: dict[str, tuple[Any, float]] = {}

# Coordenadas representativas por zona de generación renovable española
ZONE_COORDS = {
    "norte":    {"lat": 43.26, "lon": -2.93},   # Bilbao (eólica norte)
    "sur":      {"lat": 37.39, "lon": -5.99},   # Sevilla (solar sur)
    "levante":  {"lat": 39.47, "lon": -0.38},   # Valencia (solar levante)
    "interior": {"lat": 41.65, "lon": -0.88},   # Zaragoza (eólica interior)
    "galicia":  {"lat": 42.88, "lon": -8.54},   # Santiago (eólica Galicia)
    "nordeste": {"lat": 41.38, "lon":  2.17},   # Barcelona (mix)
}

# Estaciones AEMET correspondientes por zona
ZONE_AEMET_STATION = {
    "norte": "0076", "sur": "5514", "levante": "8025",
    "interior": "9898", "galicia": "1387", "nordeste": "0200E",
}


def _aemet_key() -> str | None:
    return os.environ.get("AEMET_API_KEY")


def _cached(key: str, build_fn):
    now = time.monotonic()
    if key in _cache and _cache[key][1] > now:
        return _cache[key][0]
    result = build_fn()
    _cache[key] = (result, now + CACHE_TTL)
    return result


# ---------------------------------------------------------------------------
# Open-Meteo fallback (sin token, siempre disponible)
# ---------------------------------------------------------------------------

def _open_meteo_wind(lat: float, lon: float, hours: int) -> dict:
    """Llama a Open-Meteo para datos de viento."""
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "windspeed_10m,winddirection_10m",
        "forecast_days": max(1, hours // 24 + 1),
        "timezone": "UTC",
    }
    resp = requests.get(OPEN_METEO_BASE, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _open_meteo_radiation(lat: float, lon: float, hours: int) -> dict:
    """Llama a Open-Meteo para radiación solar."""
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "shortwave_radiation,direct_normal_irradiance",
        "forecast_days": max(1, hours // 24 + 1),
        "timezone": "UTC",
    }
    resp = requests.get(OPEN_METEO_BASE, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Herramientas MCP
# ---------------------------------------------------------------------------

@mcp.tool()
def get_wind_forecast(zone: str = "norte", hours: int = 24) -> dict:
    """Previsión de viento para una zona de generación eólica.

    Args:
        zone:  'norte' | 'sur' | 'levante' | 'interior' | 'galicia' | 'nordeste'
        hours: Horizonte de previsión en horas (1-96)
    """
    zone = zone.lower()
    if zone not in ZONE_COORDS:
        return {"error": f"Zona desconocida: {zone}. Disponibles: {list(ZONE_COORDS)}"}
    hours = max(1, min(hours, 96))

    def build():
        coords = ZONE_COORDS[zone]
        try:
            data = _open_meteo_wind(coords["lat"], coords["lon"], hours)
            hourly = data.get("hourly", {})
            timestamps_raw = hourly.get("time", [])[:hours]
            speeds = hourly.get("windspeed_10m", [])[:hours]

            timestamps = [
                datetime.fromisoformat(t).replace(tzinfo=timezone.utc)
                for t in timestamps_raw
            ]
            # Conversión velocidad → % capacidad eólica instalada (curva de potencia simplificada)
            # Velocidad de arranque: 3 m/s, nominal: 12 m/s, corte: 25 m/s
            capacity_pct = []
            for v_kmh in speeds:
                v_ms = v_kmh / 3.6
                if v_ms < 3 or v_ms > 25:
                    pct = 0.0
                elif v_ms >= 12:
                    pct = 100.0
                else:
                    pct = ((v_ms - 3) / 9) ** 3 * 100
                capacity_pct.append(round(pct, 1))

            return WindForecast(
                zone=zone,
                forecast_hours=len(timestamps),
                timestamps=timestamps,
                wind_speed_ms=[round(v / 3.6, 2) for v in speeds],
                generation_capacity_pct=capacity_pct,
            ).model_dump(mode="json")
        except Exception as exc:
            return {"error": f"Error obteniendo viento para {zone}: {exc}"}

    return _cached(f"wind_{zone}_{hours}", build)


@mcp.tool()
def get_solar_radiation(zone: str = "sur", hours: int = 24) -> dict:
    """Previsión de radiación solar para una zona de generación fotovoltaica.

    Args:
        zone:  'norte' | 'sur' | 'levante' | 'interior' | 'galicia' | 'nordeste'
        hours: Horizonte de previsión en horas (1-96)
    """
    zone = zone.lower()
    if zone not in ZONE_COORDS:
        return {"error": f"Zona desconocida: {zone}. Disponibles: {list(ZONE_COORDS)}"}
    hours = max(1, min(hours, 96))

    def build():
        coords = ZONE_COORDS[zone]
        try:
            data = _open_meteo_radiation(coords["lat"], coords["lon"], hours)
            hourly = data.get("hourly", {})
            timestamps_raw = hourly.get("time", [])[:hours]
            radiation = hourly.get("shortwave_radiation", [])[:hours]

            timestamps = [
                datetime.fromisoformat(t).replace(tzinfo=timezone.utc)
                for t in timestamps_raw
            ]
            # % capacidad solar instalada: rendimiento pico a ~1000 W/m²
            capacity_pct = [
                round(min(r / 1000 * 100, 100), 1) for r in radiation
            ]
            return RadiationForecast(
                zone=zone,
                forecast_hours=len(timestamps),
                timestamps=timestamps,
                radiation_wm2=[round(r, 1) for r in radiation],
                generation_capacity_pct=capacity_pct,
            ).model_dump(mode="json")
        except Exception as exc:
            return {"error": f"Error obteniendo radiación para {zone}: {exc}"}

    return _cached(f"solar_{zone}_{hours}", build)


@mcp.tool()
def get_aemet_alerts() -> list[dict]:
    """Alertas meteorológicas activas de AEMET que pueden afectar la generación renovable."""
    def build():
        api_key = _aemet_key()
        if not api_key:
            return [{"warning": "AEMET_API_KEY no definida. Sin alertas disponibles."}]

        try:
            meta_resp = requests.get(
                f"{AEMET_BASE}/avisos_cap/ultimoelaborado/area/P",
                params={"api_key": api_key},
                timeout=30,
            )
            meta_resp.raise_for_status()
            meta = meta_resp.json()

            if meta.get("estado") != 200 or not meta.get("datos"):
                return []

            data_resp = requests.get(meta["datos"], timeout=60)
            data_resp.raise_for_status()
            raw_alerts = data_resp.json()

            alerts: list[dict] = []
            for alert in raw_alerts[:20]:  # limitar a 20 alertas
                phenomenon = alert.get("evento", "UNKNOWN").upper()
                nivel = alert.get("nivel", "YELLOW").upper()
                if nivel not in ("YELLOW", "ORANGE", "RED"):
                    nivel = "YELLOW"

                try:
                    model_alert = MeteoAlert(
                        alert_id=str(alert.get("id", "unknown")),
                        zone=alert.get("area", "desconocida"),
                        phenomenon=phenomenon,
                        severity=nivel,
                        start=datetime.fromisoformat(
                            alert.get("fechaIni", datetime.now(timezone.utc).isoformat())
                        ),
                        end=datetime.fromisoformat(
                            alert.get("fechaFin", datetime.now(timezone.utc).isoformat())
                        ),
                        description=alert.get("descripcion", ""),
                    )
                    alerts.append(model_alert.model_dump(mode="json"))
                except Exception:
                    continue

            return alerts
        except Exception as exc:
            return [{"error": f"Error obteniendo alertas AEMET: {exc}"}]

    return _cached("aemet_alerts", build)


if __name__ == "__main__":
    mcp.run()
