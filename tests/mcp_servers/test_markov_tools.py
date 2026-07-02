import json
import numpy as np
import pytest
from mcp_servers import ml_models_server as srv


@pytest.fixture(autouse=True)
def _fake_model(tmp_path, monkeypatch):
    from ml import grid_state_tracker as gst
    from tools import markov_state_tracker as mst
    A = gst.blend_with_prior(np.eye(5) * 100)
    out = tmp_path / "markov_state.npz"
    np.savez(out, transition_matrix=A,
             thresholds=np.array({"p90_cp": .8, "p97_cp": .95, "p90_ren": 80., "p85_bal": 6000.}, dtype=object),
             state_counts=np.ones(5))
    monkeypatch.setattr(mst, "MODEL_PATH", out)
    mst._reset_cache()


def test_get_grid_state_shape():
    out = srv._get_grid_state_impl({"composite_pressure": 0.1, "renewables_pct": 30, "balance_mw": 0})
    assert set(out) == {"state", "label", "confidence", "trajectory_6h"}
    assert out["state"] == 0
    assert len(out["trajectory_6h"]) == 6
    # Defensive JSON serialization: all types must be native Python (int, float, str)
    json.dumps(out)  # raises TypeError if any numpy types leak
    assert isinstance(out["state"], int)
    assert isinstance(out["confidence"], float)
    assert all(isinstance(s, int) for s in out["trajectory_6h"])


def test_markov_trajectory_probs_in_range():
    out = srv._get_markov_trajectory_impl(
        {"composite_pressure": 0.99, "renewables_pct": 90, "balance_mw": 9000}, horizon_hours=6)
    assert 0.0 <= out["prob_reach_s3"] <= 1.0
    assert 0.0 <= out["prob_reach_s4"] <= 1.0
    assert len(out["trajectory"]) == 6
    # Defensive JSON serialization: all types must be native Python
    json.dumps(out)  # raises TypeError if any numpy types leak
    assert all(isinstance(s, int) for s in out["trajectory"])
    assert isinstance(out["prob_reach_s3"], float)
    assert isinstance(out["prob_reach_s4"], float)
