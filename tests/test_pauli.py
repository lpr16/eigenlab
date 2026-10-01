import numpy as np

from eigenlab.pauli import X, Y, Z, expectation, pauli
from eigenlab.states import PLUS, ZERO


def test_pauli_algebra():
    assert np.allclose(X @ X, np.eye(2))
    assert np.allclose(X @ Y, 1j * Z)


def test_expectation_of_z_on_computational_basis():
    assert expectation(Z, ZERO) == 1.0
    assert abs(expectation(Z, PLUS)) < 1e-12


def test_two_qubit_pauli_string():
    op = pauli("XZ")
    assert op.shape == (4, 4)
    assert np.allclose(op @ op, np.eye(4))
