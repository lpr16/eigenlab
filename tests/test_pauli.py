import numpy as np

from eigenlab.pauli import (
    X,
    Y,
    Z,
    apply_pauli,
    conjugate_clifford,
    conjugate_pauli,
    expectation,
    pauli,
    pauli_anti,
    pauli_comm,
    pauli_mult,
)
from eigenlab.states import ONE, PLUS, ZERO, basis


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


def test_apply_pauli_signed_permutation(monkeypatch):
    # Phase 2: A Pauli as a signed permutation
    import itertools
    import pytest
    import eigenlab.hamiltonian
    import eigenlab.pauli
    from eigenlab.hamiltonian import Hamiltonian, term_expectation, to_matrix

    # 1. One-qubit checks
    assert np.allclose(apply_pauli("Z", ZERO), ZERO, atol=1e-12)
    assert np.allclose(apply_pauli("Z", ONE), -ONE, atol=1e-12)
    psi_1q = np.array([0.6, 0.8j], dtype=complex)
    assert np.allclose(apply_pauli("X", psi_1q), np.array([0.8j, 0.6], dtype=complex), atol=1e-12)
    assert np.allclose(apply_pauli("Y", ZERO), 1j * ONE, atol=1e-12)
    assert np.allclose(apply_pauli("Y", ONE), -1j * ZERO, atol=1e-12)

    # 2. Every label of length 2 and 3, and every computational basis ket, matches pauli(label) @ ket to 1e-12
    chars = ["I", "X", "Y", "Z"]
    for n in [2, 3]:
        for tup in itertools.product(chars, repeat=n):
            lbl = "".join(tup)
            p_mat = pauli(lbl)
            for idx in range(1 << n):
                ket = basis(idx, n)
                assert np.allclose(apply_pauli(lbl, ket), p_mat @ ket, atol=1e-12)

    # 3. Patch eigenlab.pauli.pauli and eigenlab.hamiltonian.pauli so they raise,
    # then check that expectation of "Z" on basis(0, 1) and term_expectation still return +1
    def _forbidden_pauli(*args, **kwargs):
        raise RuntimeError("pauli() matrix builder must not be called")

    monkeypatch.setattr(eigenlab.pauli, "pauli", _forbidden_pauli)
    monkeypatch.setattr(eigenlab.hamiltonian, "pauli", _forbidden_pauli)

    assert expectation("Z", basis(0, 1)) == 1.0
    assert term_expectation([(1.0, "Z")], basis(0, 1)) == 1.0
    assert Hamiltonian([(1.0, "Z")]).expectation(basis(0, 1)) == 1.0

    # 4. Nine qubits: apply_pauli("Z" + "I"*8, basis(0, 9)) equals that basis ket,
    # and to_matrix of the same one-term Hamiltonian still raises ValueError
    lbl_9 = "Z" + "I" * 8
    ket_9 = basis(0, 9)
    assert np.array_equal(apply_pauli(lbl_9, ket_9), ket_9)
    h_9 = Hamiltonian([(1.0, lbl_9)])
    assert h_9.expectation(ket_9) == 1.0
    with pytest.raises(ValueError, match="n <= 8"):
        to_matrix(h_9)


def test_conjugate_pauli():
    # Phase 3: Conjugation by a Pauli
    import itertools

    # 1. XZX = -Z and YXY = -X on one qubit
    phase_xzx, lbl_xzx = conjugate_pauli("Z", "X")
    assert phase_xzx == -1.0
    assert lbl_xzx == "Z"

    phase_yxy, lbl_yxy = conjugate_pauli("X", "Y")
    assert phase_yxy == -1.0
    assert lbl_yxy == "X"

    # 2. For every pair of length-2 labels, matches P @ Q @ P from matrices to 1e-12
    chars = ["I", "X", "Y", "Z"]
    labels_2 = ["".join(t) for t in itertools.product(chars, repeat=2)]
    for q_lbl in labels_2:
        q_mat = pauli(q_lbl)
        for p_lbl in labels_2:
            p_mat = pauli(p_lbl)
            phase, conj_lbl = conjugate_pauli(q_lbl, p_lbl)
            assert np.allclose(phase * pauli(conj_lbl), p_mat @ q_mat @ p_mat, atol=1e-12)


def test_conjugate_clifford():
    # Phase 4: Hadamard and S
    import itertools

    H_mat = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    S_mat = np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=complex)
    I_mat = np.eye(2, dtype=complex)

    # Single-qubit checks
    assert conjugate_clifford("X", "H", 0) == (1.0, "Z")
    assert conjugate_clifford("Z", "H", 0) == (1.0, "X")
    assert conjugate_clifford("Y", "H", 0) == (-1.0, "Y")

    assert conjugate_clifford("X", "S", 0) == (1.0, "Y")
    assert conjugate_clifford("Y", "S", 0) == (-1.0, "X")
    assert conjugate_clifford("Z", "S", 0) == (1.0, "Z")

    # For every length-2 label, both qubits, and both gates, matches matrix conjugation U P U† to 1e-12
    chars = ["I", "X", "Y", "Z"]
    labels_2 = ["".join(t) for t in itertools.product(chars, repeat=2)]
    gate_mats = {"H": H_mat, "S": S_mat}

    for lbl in labels_2:
        p_mat = pauli(lbl)
        for gate_name, g_mat in gate_mats.items():
            for q in [0, 1]:
                u_2q = np.kron(g_mat, I_mat) if q == 0 else np.kron(I_mat, g_mat)
                expected = u_2q @ p_mat @ u_2q.conj().T
                phase, out_lbl = conjugate_clifford(lbl, gate_name, q)
                assert np.allclose(phase * pauli(out_lbl), expected, atol=1e-12)
