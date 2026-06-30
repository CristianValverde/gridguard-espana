import numpy as np
import pytest
from tools import markov_state_tracker as mst


def test_compute_state_low_pressure_is_normal(tmp_path, monkeypatch):
    # Forzar un .npz de prueba con umbrales conocidos
    from ml import grid_state_tracker as gst
    thr = {"p90_cp": 0.8, "p97_cp": 0.95, "p90_ren": 80.0, "p85_bal": 6000.0}
    A = gst.blend_with_prior(np.eye(5) * 100)
    out = tmp_path / "markov_state.npz"
    np.savez(out, transition_matrix=A, thresholds=np.array(thr, dtype=object),
             state_counts=np.ones(5))
    monkeypatch.setattr(mst, "MODEL_PATH", out)
    mst._reset_cache()

    s = mst.compute_state({"composite_pressure": 0.1, "renewables_pct": 30, "balance_mw": 100})
    assert s == 0


def test_get_trajectory_length_and_range(tmp_path, monkeypatch):
    from ml import grid_state_tracker as gst
    thr = {"p90_cp": 0.8, "p97_cp": 0.95, "p90_ren": 80.0, "p85_bal": 6000.0}
    A = gst.blend_with_prior(np.eye(5) * 100)
    out = tmp_path / "markov_state.npz"
    np.savez(out, transition_matrix=A, thresholds=np.array(thr, dtype=object),
             state_counts=np.ones(5))
    monkeypatch.setattr(mst, "MODEL_PATH", out)
    mst._reset_cache()

    traj = mst.get_trajectory(current_state=1, horizon_hours=6)
    assert len(traj) == 6
    assert all(0 <= s <= 4 for s in traj)


def test_prob_reach_severity_is_between_0_and_1(tmp_path, monkeypatch):
    from ml import grid_state_tracker as gst
    out = tmp_path / "markov_state.npz"
    A = gst.blend_with_prior(np.eye(5) * 100)
    np.savez(out, transition_matrix=A,
             thresholds=np.array({"p90_cp": .8, "p97_cp": .95, "p90_ren": 80., "p85_bal": 6000.}, dtype=object),
             state_counts=np.ones(5))
    monkeypatch.setattr(mst, "MODEL_PATH", out)
    mst._reset_cache()

    p = mst.prob_reach_severity(current_state=2, target=3, horizon_hours=6)
    assert 0.0 <= p <= 1.0
