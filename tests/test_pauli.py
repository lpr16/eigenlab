import numpy as np

from eigenlab.pauli import X, Y, Z, expectation, pauli, pauli_mult
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


def test_pauli_multiplication_without_matrix():
    # Phase 5: Check XY = iZ, XZ = -iY, YZ = iX
    phase_xy, label_xy = pauli_mult("X", "Y")
    assert phase_xy == 1j
    assert label_xy == "Z"

    phase_xz, label_xz = pauli_mult("X", "Z")
    assert phase_xz == -1j
    assert label_xz == "Y"

    phase_yz, label_yz = pauli_mult("Y", "Z")
    assert phase_yz == 1j
    assert label_yz == "X"

    # Phases are strictly in {1, -1, i, -i}
    allowed_phases = {1.0 + 0j, -1.0 + 0j, 1j, -1j}

    # Every length-2 pair matches pauli(a) @ pauli(b)
    import itertools

    chars = ["I", "X", "Y", "Z"]
    length_2_labels = ["".join(p) for p in itertools.product(chars, repeat=2)]
    for a in length_2_labels:
        for b in length_2_labels:
            phase, prod_label = pauli_mult(a, b)
            assert phase in allowed_phases
            # Must match explicit Kronecker matrix product
            expected_matrix = pauli(a) @ pauli(b)
            actual_matrix = phase * pauli(prod_label)
            assert np.allclose(actual_matrix, expected_matrix)
