import numpy as np

from eigenlab.pauli import X, Y, Z, expectation, pauli, pauli_anti, pauli_comm, pauli_mult
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


def test_pauli_comm_and_anti():
    # Phase 1: Commutator and anticommutator
    from eigenlab.hamiltonian import Hamiltonian

    # 1. [X, Z] = XZ - ZX = -2i Y, and {X, Z} = 0
    comm_phase, comm_label = pauli_comm("X", "Z")
    assert comm_phase == -2.0j
    assert comm_label == "Y"
    assert np.allclose(comm_phase * pauli(comm_label), X @ Z - Z @ X, atol=1e-12)

    anti_phase, _ = pauli_anti("X", "Z")
    assert anti_phase == 0.0

    # 2. [XX, YY] = 0 on two qubits (commute), {XX, YY} = -2 ZZ
    comm_xx_yy_phase, _ = pauli_comm("XX", "YY")
    assert comm_xx_yy_phase == 0.0

    anti_xx_yy_phase, anti_xx_yy_label = pauli_anti("XX", "YY")
    assert anti_xx_yy_phase == -2.0
    assert anti_xx_yy_label == "ZZ"

    # 3. Term-list commutator [H1, H2] = H1 @ H2 - H2 @ H1 matches matrix commutator of XI and ZI to 1e-12
    h1 = Hamiltonian([(1.0, "XI")])
    h2 = Hamiltonian([(1.0, "ZI")])
    term_comm = (h1 @ h2 - h2 @ h1).simplify()

    mat_xi = pauli("XI")
    mat_zi = pauli("ZI")
    mat_comm = mat_xi @ mat_zi - mat_zi @ mat_xi

    assert np.allclose(term_comm.to_matrix(), mat_comm, atol=1e-12)
