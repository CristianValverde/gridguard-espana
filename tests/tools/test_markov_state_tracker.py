import numpy as np
import pytest
from ml.incidents import S0_NORMAL, S1_STRESS, S2_NOTABLE
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


def test_compute_state_s2_priority_over_s1(tmp_path, monkeypatch):
    """Test S1/S2 branches and verify S2-before-S1 priority (cp>p97 wins over S1 conditions)."""
    from ml import grid_state_tracker as gst

    thr = {"p90_cp": 0.8, "p97_cp": 0.95, "p90_ren": 80.0, "p85_bal": 6000.0}
    A = gst.blend_with_prior(np.eye(5) * 100)
    out = tmp_path / "markov_state.npz"
    np.savez(out, transition_matrix=A, thresholds=np.array(thr, dtype=object),
             state_counts=np.ones(5))
    monkeypatch.setattr(mst, "MODEL_PATH", out)
    mst._reset_cache()

    # S2: cp > p97_cp (0.95) returns S2_NOTABLE even if other S1 conditions not met
    assert mst.compute_state({"composite_pressure": 0.96, "renewables_pct": 10, "balance_mw": 0}) == S2_NOTABLE

    # S1: cp > p90_cp (0.8) but < p97_cp (0.95) returns S1_STRESS
    assert mst.compute_state({"composite_pressure": 0.85, "renewables_pct": 10, "balance_mw": 0}) == S1_STRESS

    # S1: renewables > p90_ren (80) AND |balance| > p85_bal (6000) returns S1_STRESS
    assert mst.compute_state({"composite_pressure": 0.1, "renewables_pct": 90, "balance_mw": 7000}) == S1_STRESS

    # S0: all conditions below thresholds
    assert mst.compute_state({"composite_pressure": 0.1, "renewables_pct": 30, "balance_mw": 100}) == S0_NORMAL

    # Critical: S2 takes priority over S1 when cp > p97
    # (cp=0.96 > p97=0.95, so S2, even though balance condition would otherwise trigger S1)
    assert mst.compute_state({"composite_pressure": 0.96, "renewables_pct": 90, "balance_mw": 7000}) == S2_NOTABLE
