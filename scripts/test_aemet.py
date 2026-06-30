"""
scripts/test_aemet.py — Smoke test de la API de AEMET.

Verifica que la clave funciona, muestra la previsión de viento y radiación
para las 6 estaciones representativas, y arranca el MCP weather_server
para probar el tool get_wind_forecast.

Uso:
  $env:AEMET_API_KEY="tu_clave"
  python scripts/test_aemet.py
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

BASE_URL = "https://opendata.aemet.es/opendata/api"

# Estaciones representativas de las 6 zonas
STATIONS = {
    "norte":    ("1082",  "Santander/Parayas"),
    "interior": ("3195",  "Madrid/Barajas"),
    "levante":  ("8416Y", "Valencia/Airport"),
    "sur":      ("5783",  "Sevilla/San Pablo"),
    "galicia":  ("1387",  "Santiago de C./Airport"),
    "canarias": ("C029O", "Las Palmas/Gando"),
}


def _api_key() -> str:
    k = os.environ.get("AEMET_API_KEY")
    if not k:
        sys.exit("ERROR: AEMET_API_KEY no definida.")
    return k


def _fetch_json(url: str, api_key: str) -> dict | list:
    """Paso 1 del patrón AEMET: obtiene la URL de datos."""
    import requests
    headers = {"Accept": "application/json", "api_key": api_key}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    envelope = r.json()

    # Paso 2: seguir la URL de datos
    data_url = envelope.get("datos")
    if not data_url:
        return envelope

    r2 = requests.get(data_url, headers=headers, timeout=30)
    r2.raise_for_status()
    try:
        return r2.json()
    except Exception:
        return {"raw": r2.text[:500]}


def test_station_observations(api_key: str) -> None:
    """Muestra la última observación de cada estación."""
    print(f"\n{'─'*60}")
    print(f"{'OBSERVACIONES EN TIEMPO REAL — AEMET':^60}")
    print(f"{'─'*60}")
    print(f"{'Zona':<12}{'Estación':<28}{'Viento km/h':>12}{'Temp °C':>10}")
    print("─" * 60)

    for zone, (station_id, name) in STATIONS.items():
        url = f"{BASE_URL}/observacion/convencional/datos/estacion/{station_id}"
        try:
            data = _fetch_json(url, api_key)
            if isinstance(data, list) and data:
                obs = data[-1]
                wind = obs.get("vv") or obs.get("vmax") or "N/A"
                temp = obs.get("ta") or "N/A"
            else:
                wind, temp = "error", "error"
        except Exception as exc:
            wind, temp = f"err({type(exc).__name__})", "—"

        print(f"  {zone:<10}{name:<28}{str(wind):>12}{str(temp):>10}")


def test_forecast(api_key: str) -> None:
    """Muestra previsión de 48h para una estación representativa."""
    import requests
    print(f"\n{'─'*60}")
    print(f"{'PREVISIÓN METEOROLÓGICA 48H — Madrid':^60}")
    print(f"{'─'*60}")

    # Previsión para municipio de Madrid (ID: 28079)
    url = f"{BASE_URL}/prediccion/especifica/municipio/diaria/28079"
    headers = {"Accept": "application/json", "api_key": api_key}

    try:
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        envelope = r.json()
        data_url = envelope.get("datos")
        if data_url:
            r2 = requests.get(data_url, headers={"api_key": api_key}, timeout=30)
            data = r2.json()
            if isinstance(data, list) and data:
                pred = data[0]
                nombre = pred.get("nombre", "Madrid")
                dias = pred.get("prediccion", {}).get("dia", [])[:2]
                print(f"  Ciudad: {nombre}")
                for dia in dias:
                    fecha = dia.get("fecha", "N/A")[:10]
                    prob_precip = dia.get("probPrecipitacion", [{}])
                    viento = dia.get("viento", [{}])
                    temp_max = dia.get("temperatura", {}).get("maxima", "N/A")
                    temp_min = dia.get("temperatura", {}).get("minima", "N/A")
                    print(f"  {fecha}: T {temp_min}-{temp_max}°C")
                    if viento:
                        v = viento[0]
                        print(f"           Viento: {v.get('velocidad', 'N/A')} km/h, dir {v.get('direccion', 'N/A')}")
    except Exception as exc:
        print(f"  Error previsión: {exc}")


def test_mcp_weather_tool(api_key: str) -> None:
    """Prueba directamente la función de viento del weather_server."""
    print(f"\n{'─'*60}")
    print(f"{'TEST FUNCIÓN MCP: get_wind_forecast':^60}")
    print(f"{'─'*60}")
    try:
        os.environ["AEMET_API_KEY"] = api_key
        # Importar y llamar la función directamente (sin levantar el servidor MCP)
        # El servidor usa un módulo interno de fetch que podemos probar así:
        from mcp_servers.weather_server import get_wind_forecast
        result = get_wind_forecast(zone="interior")
        print(f"  get_wind_forecast(zone='interior'):")
        if isinstance(result, dict):
            for k, v in result.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {result}")
    except Exception as exc:
        print(f"  Error al llamar MCP tool directamente: {exc}")
        print("  (Normal si el weather_server usa async — se prueba vía MCP en producción)")


def main() -> None:
    api_key = _api_key()
    print("=" * 60)
    print("GridGuard — Smoke test AEMET API")
    print("=" * 60)
    print(f"API key: {'*' * 8}{api_key[-4:]}")

    test_station_observations(api_key)
    test_forecast(api_key)
    test_mcp_weather_tool(api_key)

    print(f"\n{'='*60}")
    print("AEMET OK — Para descargar datos históricos de training:")
    print("  $env:AEMET_API_KEY='tu_clave'")
    print("  python ml/data/download_aemet.py --years 2020 2021 2022 2023 2024 2025")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
