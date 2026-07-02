"""Paridad de IDs ESIOS del poller con download_esios (fuente verificada en producción)."""
from __future__ import annotations

from ml.data.download_esios import INDICATORS
from tools import ree_poller as rp


def test_generation_ids_verified():
    for key, ind_id in rp._GEN_INDICATORS.items():
        assert ind_id == INDICATORS["generation"][key], (
            f"{key}: poller={ind_id}, verificado={INDICATORS['generation'][key]}"
        )


def test_interchange_ids_verified():
    for key, ind_id in rp._INTERCHANGE_INDICATORS.items():
        assert ind_id == INDICATORS["interchange"][key], (
            f"{key}: poller={ind_id}, verificado={INDICATORS['interchange'][key]}"
        )


def test_demand_real_id_verified():
    assert rp._DEMAND_INDICATORS["demand_real_mw"] == INDICATORS["demand"]["demand_real_mw"]
