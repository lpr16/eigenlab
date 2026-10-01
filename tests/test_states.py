import numpy as np

from eigenlab.states import bell, probabilities


def test_phi_plus_is_correlated():
    probs = probabilities(bell("phi+"))
    assert np.allclose(probs, [0.5, 0.0, 0.0, 0.5])


def test_psi_minus_is_anticorrelated():
    probs = probabilities(bell("psi-"))
    assert np.allclose(probs, [0.0, 0.5, 0.5, 0.0])
