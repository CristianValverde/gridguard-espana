"""
eval/conftest.py — Fixtures compartidas entre evals de agentes.

Movidas aquí desde eval/test_outputs.py y eval/test_trajectories.py
(Task 8: reestructura de eval/ por agente).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

GOLDEN_DIR = Path("eval/golden_dataset")


@pytest.fixture
def sample_snapshot():
    return {
        "timestamp": "2025-04-28T10:30:00Z",
        "region": "peninsular",
        "demand_mw": 30200,
        "generation_by_tech": {
            "nuclear_mw": 7100, "hydro_mw": 800,
            "wind_mw": 14200, "solar_pv_mw": 13500,
            "solar_thermal_mw": 2800, "gas_cc_mw": 0,
            "coal_mw": 0, "cogeneration_mw": 1800,
        },
        "interchange_flows": {
            "es_fr_mw": -1800, "es_pt_mw": -310, "es_ma_mw": 60,
        },
        "reserve_mw": 1240,
    }


@pytest.fixture
def blackout_scenario() -> dict:
    return json.loads((GOLDEN_DIR / "blackout_apr_2025.json").read_text(encoding="utf-8"))


@pytest.fixture
def normal_scenario() -> dict:
    return json.loads((GOLDEN_DIR / "normal_baseline.json").read_text(encoding="utf-8"))


@pytest.fixture
def critical_snapshot(blackout_scenario) -> dict:
    """Snapshot más cercano al evento (T-3min)."""
    return blackout_scenario["snapshots"][-1]


@pytest.fixture
def normal_snapshot(normal_scenario) -> dict:
    return normal_scenario["snapshots"][0]
