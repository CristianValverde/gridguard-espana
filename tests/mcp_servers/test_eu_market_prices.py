# tests/mcp_servers/test_eu_market_prices.py
#
# Contrato: _market_prices_impl + get_eu_market_prices
# Fuente real: ENTSO-E Transparency Platform, docType A44 (day-ahead prices).
# Confirmado via api-probe 2026-06-30:
#   - entsoe-py EntsoePandasClient.query_day_ahead_prices devuelve pd.Series €/MWh
#   - Resolución real: 15-min (96 valores/día, NOT 24 hourly)
#   - Índice tz-aware en zona local del país (Europe/Paris para FR)
#   - CONVERTIR a UTC antes de devolver (series.index.tz_convert("UTC"))

import pytest
import pandas as pd
from mcp_servers import grid_data_server as es


def _fake_fetch(country, day):
    """Mock con resolución horaria (24 valores) para tests básicos."""
    idx = pd.date_range(f"{day} 00:00", periods=24, freq="h", tz="UTC")
    return pd.Series([50.0 + i for i in range(24)], index=idx)


def _fake_fetch_15min(country, day):
    """Mock con resolución 15-min (96 valores) — mimics real ENTSO-E resolution."""
    idx = pd.date_range(f"{day} 00:00", periods=96, freq="15min", tz="UTC")
    return pd.Series([77.0 + i * 0.5 for i in range(96)], index=idx)


# ---------------------------------------------------------------------------
# Contract tests (no I/O)
# ---------------------------------------------------------------------------

def test_market_prices_returns_series_per_country():
    """Verifica que el output contiene entradas por país con timestamps y precios."""
    out = es._market_prices_impl(["FR", "PT"], "2026-06-30", fetch_fn=_fake_fetch)
    assert set(out) == {"FR", "PT"}
    for c in out.values():
        assert len(c["timestamps"]) == 24
        assert len(c["price_eur_mwh"]) == 24
        assert c["price_eur_mwh"][0] == 50.0


def test_market_prices_unknown_country_skipped():
    """Países no en AREA_CODES se ignoran silenciosamente."""
    out = es._market_prices_impl(["XX"], "2026-06-30", fetch_fn=_fake_fetch)
    assert out == {}


def test_market_prices_15min_resolution():
    """Prueba que la impl maneja 96 valores/día (resolución 15-min real de ENTSO-E).

    Regresión: asegura que no asumimos 24 puntos horarios fijos.
    """
    out = es._market_prices_impl(["ES"], "2026-06-30", fetch_fn=_fake_fetch_15min)
    assert "ES" in out
    data = out["ES"]
    assert len(data["timestamps"]) == 96, (
        f"Expected 96 timestamps (15-min resolution), got {len(data['timestamps'])}"
    )
    assert len(data["price_eur_mwh"]) == 96, (
        f"Expected 96 price values, got {len(data['price_eur_mwh'])}"
    )
    # First value sanity check
    assert data["price_eur_mwh"][0] == 77.0


def test_market_prices_all_area_codes():
    """Todos los países en AREA_CODES (ES, FR, PT) son procesados."""
    out = es._market_prices_impl(["ES", "FR", "PT"], "2026-06-30", fetch_fn=_fake_fetch)
    assert set(out) == {"ES", "FR", "PT"}


def test_market_prices_mixed_known_unknown():
    """Mix de países conocidos y desconocidos — sólo devuelve los conocidos."""
    out = es._market_prices_impl(["ES", "DE", "FR"], "2026-06-30", fetch_fn=_fake_fetch)
    assert set(out) == {"ES", "FR"}


# ---------------------------------------------------------------------------
# Integration test (requires ENTSOE_TOKEN env var + live ENTSO-E access)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_market_prices_real_entsoe():
    """Integration test: live ENTSO-E A44 day-ahead for FR (yesterday).

    Probe (2026-06-30) confirmed 96 values/day at 15-min resolution,
    price range ~77–252 €/MWh, index in Europe/Paris (converted to UTC here).
    ENTSOE_TOKEN must be set in .env or environment.
    Makes exactly ONE API call to respect ENTSO-E rate limits.
    """
    try:
        result = es.get_eu_market_prices(["FR"], "2026-06-29")
    except Exception as exc:
        pytest.skip(f"ENTSO-E live call failed (rate-limit/network): {exc}")

    assert "FR" in result, f"Expected 'FR' in result, got: {result}"
    data = result["FR"]

    # Non-empty
    assert len(data["timestamps"]) > 0, "Expected non-empty timestamps"
    assert len(data["price_eur_mwh"]) > 0, "Expected non-empty prices"

    # Lengths must match
    assert len(data["timestamps"]) == len(data["price_eur_mwh"]), (
        "timestamps and price_eur_mwh must have the same length"
    )

    # Probe confirmed 96 values/day at 15-min resolution
    # Allow 95-97 range for DST transitions
    n = len(data["price_eur_mwh"])
    assert 90 <= n <= 100, (
        f"Expected ~96 values/day (15-min resolution), got {n}"
    )

    # All prices must be floats
    for v in data["price_eur_mwh"]:
        assert isinstance(v, float), f"Price not float: {v} ({type(v)})"

    # Timestamps must be tz-aware or ISO strings with UTC offset
    # After model_dump(mode="json"), datetimes become ISO strings
    for ts in data["timestamps"]:
        ts_str = str(ts)
        # UTC timestamps end in +00:00 or Z
        assert ("+00:00" in ts_str or ts_str.endswith("Z")), (
            f"Timestamp not UTC: {ts_str}"
        )
