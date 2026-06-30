# tests/mcp_servers/test_weather_forecast_24h.py
#
# Contrato: _forecast_24h_impl + get_weather_forecast_24h
# Fuente real: Open-Meteo (AEMET horaria no da radiación + rate-limit 429).
# Confirmado via api-probe 2026-06-30.

import pytest
from mcp_servers import weather_server as ws


def _fake_fetch(zone, coords):
    # Formato confirmado por el probe; 24 entradas horarias por zona.
    return {
        "timestamps": [f"2026-06-30T{h:02d}:00:00Z" for h in range(24)],
        "wind_speed_ms": [5.0] * 24,
        "radiation_wm2": [0.0] * 6 + [300.0] * 12 + [0.0] * 6,
    }


def test_forecast_24h_returns_24h_per_zone():
    out = ws._forecast_24h_impl(["norte", "sur"], fetch_fn=_fake_fetch)
    assert set(out) == {"norte", "sur"}
    for zone_data in out.values():
        assert zone_data["forecast_hours"] == 24
        assert len(zone_data["timestamps"]) == 24
        assert len(zone_data["wind_speed_ms"]) == 24
        assert len(zone_data["radiation_wm2"]) == 24


def test_forecast_24h_unknown_zone_is_skipped():
    out = ws._forecast_24h_impl(["zona_inexistente"], fetch_fn=_fake_fetch)
    assert out == {}


@pytest.mark.integration
def test_forecast_24h_real_openmeteo():
    """Integration test: live Open-Meteo call for zone 'norte'. No token needed."""
    result = ws.get_weather_forecast_24h(["norte"])

    assert "norte" in result, f"Expected 'norte' in result, got: {result}"
    data = result["norte"]

    assert data["forecast_hours"] == 24
    assert len(data["timestamps"]) == 24, f"Expected 24 timestamps, got {len(data['timestamps'])}"
    assert len(data["wind_speed_ms"]) == 24, f"Expected 24 wind values, got {len(data['wind_speed_ms'])}"
    assert len(data["radiation_wm2"]) == 24, f"Expected 24 radiation values, got {len(data['radiation_wm2'])}"

    # Wind values must be non-negative floats
    for v in data["wind_speed_ms"]:
        assert isinstance(v, (int, float)), f"Wind value not numeric: {v}"
        assert v >= 0, f"Wind speed negative: {v}"

    # At least one radiation value > 0 (daytime, end of June in Spain)
    assert any(r > 0 for r in data["radiation_wm2"]), (
        f"Expected at least one radiation > 0, all zero. Values: {data['radiation_wm2']}"
    )
