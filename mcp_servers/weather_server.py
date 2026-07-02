"""
mcp_servers/weather_server.py — MCP: weather

Datos meteorológicos para zonas de generación renovable.
Primario: Open-Meteo (gratuito, sin token).

Herramientas expuestas:
  get_wind_forecast(zone, hours)      -> WindForecast (JSON)
  get_solar_radiation(zone, hours)    -> RadiationForecast (JSON)
  get_weather_forecast_24h(zones)     -> dict con previsión horaria a 24h

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

from schemas.models import RadiationForecast, WindForecast

mcp = FastMCP("gridguard-weather")

OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"
CACHE_TTL = 1800  # 30 min — datos meteorológicos cambian más despacio
REQUEST_TIMEOUT = 30  # segundos para llamadas HTTP a Open-Meteo

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
    resp = requests.get(OPEN_METEO_BASE, params=params, timeout=REQUEST_TIMEOUT)
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
    resp = requests.get(OPEN_METEO_BASE, params=params, timeout=REQUEST_TIMEOUT)
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


# ---------------------------------------------------------------------------
# get_weather_forecast_24h — Open-Meteo (fuente confirmada via api-probe 2026-06-30)
#
# get_weather_forecast_24h fuente: Open-Meteo (AEMET horaria no da radiación + rate-limit 429).
# Confirmado via api-probe 2026-06-30.
# GET https://api.open-meteo.com/v1/forecast
# params: latitude, longitude, hourly="wind_speed_10m,shortwave_radiation",
#         wind_speed_unit="ms", forecast_days=2, timezone="UTC"
# resp: data["hourly"]["time"] (iso8601 UTC), ["wind_speed_10m"] (m/s),
#       ["shortwave_radiation"] (W/m²) — 48 valores
# ---------------------------------------------------------------------------

def _fetch_openmeteo_hourly(zone: str, coords: dict) -> dict:
    """Transporte HTTP a Open-Meteo: viento (m/s) + radiación solar (W/m²) a 48h.

    Nota: Open-Meteo no requiere token y no tiene rate-limit.
    No se usa caché aquí para mantener el código simple (YAGNI); la TTL de 30 min
    del resto del módulo se puede añadir si el tool se llama frecuentemente.
    """
    params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "hourly": "wind_speed_10m,shortwave_radiation",
        "wind_speed_unit": "ms",
        "forecast_days": 2,
        "timezone": "UTC",
    }
    resp = requests.get(OPEN_METEO_BASE, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    # Verificar errores en Open-Meteo (HTTP 200 con body {"error": true, "reason": "..."})
    if isinstance(data, dict) and data.get("error"):
        raise ValueError(f"Open-Meteo error para zona {zone}: {data.get('reason', 'desconocido')}")

    # Verificar que la respuesta contiene la clave 'hourly'
    if "hourly" not in data:
        raise ValueError(f"Respuesta Open-Meteo sin 'hourly' para zona {zone}: {str(data)[:200]}")

    hourly = data["hourly"]
    return {
        "timestamps": hourly["time"],
        "wind_speed_ms": hourly["wind_speed_10m"],
        "radiation_wm2": hourly["shortwave_radiation"],
    }


def _forecast_24h_impl(zones: list[str], fetch_fn) -> dict:
    """Lógica pura: para cada zona conocida, obtiene 24 h de previsión vía fetch_fn.

    fetch_fn(zone, coords) -> {"timestamps": [...], "wind_speed_ms": [...], "radiation_wm2": [...]}
    Las listas se truncan a las primeras 24 entradas.
    Zonas desconocidas (no en ZONE_COORDS) se saltan silenciosamente.
    """
    out: dict = {}
    for zone in zones:
        coords = ZONE_COORDS.get(zone)
        if coords is None:
            continue
        data = fetch_fn(zone, coords)
        out[zone] = {
            "forecast_hours": 24,
            "timestamps": data["timestamps"][:24],
            "wind_speed_ms": data["wind_speed_ms"][:24],
            "radiation_wm2": data["radiation_wm2"][:24],
        }
    return out


@mcp.tool()
def get_weather_forecast_24h(zones: list[str]) -> dict:
    """Previsión meteorológica horaria a 24 h para las zonas de generación renovable indicadas.

    Fuente: Open-Meteo (sin token, gratuito).
    AEMET horaria descartada: no provee radiación solar y tiene rate-limit 429.

    Args:
        zones: Lista de zonas. Valores válidos: norte, sur, levante, interior, galicia, nordeste
    Returns:
        Dict {zona: {"forecast_hours": 24, "timestamps": [...], "wind_speed_ms": [...],
                     "radiation_wm2": [...]}} — solo zonas reconocidas.
    """
    return _forecast_24h_impl(zones, fetch_fn=_fetch_openmeteo_hourly)


if __name__ == "__main__":
    mcp.run()
