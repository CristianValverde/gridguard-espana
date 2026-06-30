import numpy as np
import pandas as pd
from ml import grid_state_tracker as gst


def test_transition_matrix_is_row_stochastic():
    states = pd.Series([0, 0, 1, 1, 0, 1, 2, 1, 0, 0])
    A = gst.estimate_transition_matrix(states, n_states=5, alpha=1.0)
    assert A.shape == (5, 5)
    np.testing.assert_allclose(A.sum(axis=1), np.ones(5), rtol=1e-9)
    assert (A >= 0).all()


def test_transition_matrix_reflects_observed_counts():
    # Secuencia 0->1 cuatro veces, nunca 0->4. Con Laplace alpha pequeño,
    # P(1|0) debe ser >> P(4|0).
    states = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    A = gst.estimate_transition_matrix(states, n_states=5, alpha=0.01)
    assert A[0, 1] > A[0, 4]
