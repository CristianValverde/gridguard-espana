"""Tests del MCP grid-data: paridad de IDs ESIOS con download_esios (fuente verificada)."""
from __future__ import annotations


def test_generation_ids_match_verified_download_esios():
    """Los IDs de generación deben ser EXACTAMENTE los verificados en producción
    (ml/data/download_esios.py, junio 2026). El bug original: nuclear=460 → ~125 GW."""
    from mcp_servers import grid_data_server as gd
    from ml.data.download_esios import INDICATORS

    verified = INDICATORS["generation"]
    for key, ind_id in gd._GEN_INDICATORS.items():
        assert key in verified, f"{key} no existe en download_esios.INDICATORS['generation']"
        assert ind_id == verified[key], (
            f"ID divergente para {key}: grid_data={ind_id}, verificado={verified[key]}"
        )


def test_entsoe_tools_present():
    """Las 4 tools ENTSO-E migran íntegras desde entsoe_server."""
    from mcp_servers import grid_data_server as gd
    for fn in ("get_cross_border_flows", "get_eu_generation",
               "get_capacity_margin", "get_eu_market_prices", "get_generation_mix"):
        assert callable(getattr(gd, fn)), f"tool ausente: {fn}"


def test_area_codes_preserved():
    from mcp_servers import grid_data_server as gd
    assert set(gd.AREA_CODES) == {"ES", "FR", "PT"}
