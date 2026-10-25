import numpy as np

import pytest

from eigenlab.states import (
    ONE,
    PLUS,
    PLUS_I,
    ZERO,
    basis,
    bell,
    bloch,
    chsh_operator,
    concurrence,
    cyclic_shift,
    density,
    entanglement_entropy,
    entanglement_of_formation,
    ghz,
    grover_once,
    helstrom,
    negativity,
    partial_trace,
    probabilities,
    purify,
    purity,
    qft_matrix,
    schmidt_spectrum,
    shannon_entropy,
    swap_test,
    w_state,
)


def test_phi_plus_is_correlated():
    probs = probabilities(bell("phi+"))
    assert np.allclose(probs, [0.5, 0.0, 0.0, 0.5])


def test_psi_minus_is_anticorrelated():
    probs = probabilities(bell("psi-"))
    assert np.allclose(probs, [0.0, 0.5, 0.5, 0.0])


def test_bloch_one_qubit_states():
    # Phase 1: |0⟩ → (0, 0, 1), |+⟩ → (1, 0, 0), |+i⟩ → (0, 1, 0)
    assert np.allclose(bloch(ZERO), [0.0, 0.0, 1.0])
    assert np.allclose(bloch(PLUS), [1.0, 0.0, 0.0])
    assert np.allclose(bloch(PLUS_I), [0.0, 1.0, 0.0])

    # Pure states lie on the unit sphere S^2: r_x^2 + r_y^2 + r_z^2 = 1
    # For |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ) sin(θ/2)|1⟩:
    # r = (sin(θ)cos(φ), sin(θ)sin(φ), cos(θ))
    theta, phi = 0.7, 1.2
    psi = np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)], dtype=complex)
    r = bloch(psi)
    assert np.isclose(np.linalg.norm(r), 1.0)
    expected_r = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    assert np.allclose(r, expected_r)


def test_bloch_rejects_multiqubit():
    with pytest.raises(ValueError, match="one-qubit"):
        bloch(np.kron(ZERO, ZERO))


def test_density_and_purity():
    # Phase 2: Every current ket has purity 1
    current_kets = [ZERO, ONE, PLUS, PLUS_I, bell("phi+"), bell("phi-"), bell("psi+"), bell("psi-"), basis(2, 3)]
    for ket in current_kets:
        rho = density(ket)
        # Tr(ρ) == 1 and Tr(ρ²) == 1 for pure states
        assert np.isclose(np.trace(rho), 1.0)
        assert np.isclose(purity(rho), 1.0)
        assert np.isclose(purity(ket), 1.0)

    # The equal mixture of |0⟩ and |1⟩ has purity 1/2:
    # ρ_mix = 1/2 |0⟩⟨0| + 1/2 |1⟩⟨1| = diag(1/2, 1/2)
    # Tr(ρ_mix²) = (1/2)² + (1/2)² = 1/4 + 1/4 = 1/2
    rho_mix = 0.5 * density(ZERO) + 0.5 * density(ONE)
    assert np.isclose(purity(rho_mix), 0.5)

    # density of an unnormalized ket matches the normalized one:
    # For ket scaled by a non-zero complex factor c = (3.0 + 4.0j)
    unnormalized = (3.0 + 4.0j) * PLUS
    assert np.allclose(density(unnormalized), density(PLUS))
    assert np.isclose(purity(unnormalized), 1.0)


def test_partial_trace():
    # Phase 3:
    # 1. |00⟩ traced over qubit 1 is |0⟩⟨0| (keep qubit 0)
    ket_00 = basis(0, 2)
    rho_00 = density(ket_00)
    reduced_0 = partial_trace(rho_00, 2, keep=0)
    assert np.allclose(reduced_0, density(ZERO))
    # Also verify passing ket directly
    assert np.allclose(partial_trace(ket_00, 2, keep=0), density(ZERO))

    # 2. Either reduction of bell("phi+") is I/2
    # Traced over qubit 1 (keep 0) or traced over qubit 0 (keep 1):
    phi_plus = bell("phi+")
    red_keep_0 = partial_trace(phi_plus, 2, keep=0)
    red_keep_1 = partial_trace(phi_plus, 2, keep=1)
    i_over_2 = 0.5 * np.eye(2)
    assert np.allclose(red_keep_0, i_over_2)
    assert np.allclose(red_keep_1, i_over_2)

    # 3. Tracing both sides returns 1
    tr_both = partial_trace(phi_plus, 2, keep=[])
    assert tr_both == 1
    assert np.isclose(tr_both, 1.0)


def test_two_qubit_entanglement():
    # Phase 4:
    # All four Bell states have concurrence 1 and entropy 1 (in bits).
    bell_names = ["phi+", "phi-", "psi+", "psi-"]
    for name in bell_names:
        state = bell(name)
        # Schmidt spectrum across the middle cut: [1/√2, 1/√2]
        s = schmidt_spectrum(state)
        assert np.allclose(s, [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)])
        # Entanglement entropy: - (0.5 log2(0.5) + 0.5 log2(0.5)) = 1 bit
        assert np.isclose(entanglement_entropy(state), 1.0)
        # Concurrence: 1
        assert np.isclose(concurrence(state), 1.0)
        # Also test density matrix concurrence
        assert np.isclose(concurrence(density(state)), 1.0)

    # |00⟩, |+⟩⊗|0⟩, and a phased product state have concurrence 0 and entropy 0.
    state_00 = np.kron(ZERO, ZERO)
    assert np.isclose(concurrence(state_00), 0.0)
    assert np.isclose(entanglement_entropy(state_00), 0.0)

    state_plus_zero = np.kron(PLUS, ZERO)
    assert np.isclose(concurrence(state_plus_zero), 0.0)
    assert np.isclose(entanglement_entropy(state_plus_zero), 0.0)

    # Phased product state: (cos(θ)|0⟩ + e^(iφ) sin(θ)|1⟩) ⊗ (cos(α)|0⟩ + e^(iβ) sin(α)|1⟩)
    theta, phi_angle = 0.3, 0.9
    alpha, beta_angle = 1.1, -0.4
    q0 = np.array([np.cos(theta), np.exp(1j * phi_angle) * np.sin(theta)], dtype=complex)
    q1 = np.array([np.cos(alpha), np.exp(1j * beta_angle) * np.sin(alpha)], dtype=complex)
    phased_product = np.kron(q0, q1)
    assert np.isclose(concurrence(phased_product), 0.0)
    assert np.isclose(entanglement_entropy(phased_product), 0.0)

    # Concurrence stays in [0, 1] for arbitrary superposition:
    # |ψ⟩ = cos(t)|00⟩ + sin(t)|11⟩ with concurrence = 2|cos(t)sin(t)| = |sin(2t)| ∈ [0, 1]
    for t in np.linspace(0, np.pi, 20):
        superposed = np.array([np.cos(t), 0, 0, np.sin(t)], dtype=complex)
        c = concurrence(superposed)
        assert 0.0 <= c <= 1.0
        assert np.isclose(c, np.abs(np.sin(2 * t)))


def test_ghz_state():
    # Phase 6: GHZ
    from eigenlab.pauli import expectation_label

    # For n = 2 this is bell("phi+")
    assert np.allclose(ghz(2), bell("phi+"), atol=1e-12)

    for n in [2, 3, 4]:
        state = ghz(n)
        # 1. Entanglement entropy across the middle cut is 1 bit
        assert np.isclose(entanglement_entropy(state), 1.0, atol=1e-12)

        # 2. The one-qubit reduction has purity 1/2
        for q in range(n):
            red = partial_trace(state, n, keep=q)
            assert np.isclose(purity(red), 0.5, atol=1e-12)

        # 3. ⟨X⊗n⟩ = 1, and ⟨Z_i Z_{i+1}⟩ = 1 for each neighboring pair
        assert np.isclose(expectation_label(state, "X" * n), 1.0, atol=1e-12)
        for i in range(n - 1):
            zz_lbl = ["I"] * n
            zz_lbl[i] = "Z"
            zz_lbl[i + 1] = "Z"
            assert np.isclose(expectation_label(state, "".join(zz_lbl)), 1.0, atol=1e-12)


def test_w_state():
    # Phase 7: W state
    w = w_state()
    expected_w = (basis(1, 3) + basis(2, 3) + basis(4, 3)) / np.sqrt(3.0)
    assert np.allclose(w, expected_w, atol=1e-12)

    # Middle-cut entropy equals binary entropy of 1/3
    expected_entropy = -(1 / 3) * np.log2(1 / 3) - (2 / 3) * np.log2(2 / 3)
    assert np.isclose(entanglement_entropy(w), expected_entropy, atol=1e-12)

    # Every two-qubit reduction has concurrence 2/3
    for keep_pair in [(0, 1), (0, 2), (1, 2)]:
        red_2q = partial_trace(w, 3, keep=keep_pair)
        assert np.isclose(concurrence(red_2q), 2.0 / 3.0, atol=1e-12)


def test_purify():
    # Phase 8: Purification
    rho = np.array([[0.7, 0.2 - 0.1j], [0.2 + 0.1j, 0.3]], dtype=complex)
    psi_pur = purify(rho)
    assert psi_pur.shape == (4,)

    # 1. Tracing out qubit 1 (keep qubit 0) returns ρ
    red_0 = partial_trace(psi_pur, 2, keep=0)
    assert np.allclose(red_0, rho, atol=1e-12)

    # 2. purify of I/2 has concurrence 1
    i_over_2 = 0.5 * np.eye(2, dtype=complex)
    assert np.isclose(concurrence(purify(i_over_2)), 1.0, atol=1e-12)

    # 3. Squared Schmidt spectrum equals the eigenvalues of ρ
    s_sq = np.sort(schmidt_spectrum(psi_pur) ** 2)[::-1]
    rho_evals = np.sort(np.linalg.eigvalsh(rho))[::-1]
    assert np.allclose(s_sq, rho_evals, atol=1e-12)


def test_negativity():
    # Phase 9: Negativity
    for kind in ["phi+", "phi-", "psi+", "psi-"]:
        assert np.isclose(negativity(bell(kind)), 0.5, atol=1e-12)
        assert np.isclose(negativity(density(bell(kind))), 0.5, atol=1e-12)

    assert negativity(basis(0, 2)) == 0.0
    assert negativity(np.kron(PLUS, ZERO)) == 0.0

    # Werner state p |Ψ⁻⟩⟨Ψ⁻| + (1−p) I/4 has negativity max(0, (3p−1)/4)
    psi_minus_dm = density(bell("psi-"))
    i4 = np.eye(4, dtype=complex) / 4.0
    for p in [0.0, 1.0 / 3.0, 0.5, 1.0]:
        werner = p * psi_minus_dm + (1.0 - p) * i4
        expected = max(0.0, (3.0 * p - 1.0) / 4.0)
        val = negativity(werner)
        if p == 1.0 / 3.0:
            # Separable boundary: value is exactly 0, not a small positive number
            assert val == 0.0
        else:
            assert np.isclose(val, expected, atol=1e-12)


def test_entanglement_of_formation():
    # Phase 10: Entanglement of formation
    for kind in ["phi+", "phi-", "psi+", "psi-"]:
        assert np.isclose(entanglement_of_formation(bell(kind)), 1.0, atol=1e-12)

    assert entanglement_of_formation(basis(0, 2)) == 0.0
    assert entanglement_of_formation(np.kron(PLUS, ZERO)) == 0.0

    psi_minus_dm = density(bell("psi-"))
    i4 = np.eye(4, dtype=complex) / 4.0
    werner_p1 = 1.0 * psi_minus_dm + 0.0 * i4
    werner_p13 = (1.0 / 3.0) * psi_minus_dm + (2.0 / 3.0) * i4

    assert np.isclose(entanglement_of_formation(werner_p1), 1.0, atol=1e-12)
    assert entanglement_of_formation(werner_p13) == 0.0


def test_shannon_entropy():
    # Phase 13: Shannon entropy
    assert shannon_entropy(basis(0, 1)) == 0.0
    assert np.isclose(shannon_entropy(bell("phi+")), 1.0, atol=1e-12)
    assert np.isclose(shannon_entropy(w_state()), np.log2(3.0), atol=1e-12)


def test_helstrom():
    # Phase 14: Helstrom bound (1 + √(1 - |⟨ψ|φ⟩|²)) / 2
    # Identical states: |⟨0|0⟩|² = 1 => (1 + √(1 - 1)) / 2 = 1/2
    assert np.isclose(helstrom(ZERO, ZERO), 0.5, atol=1e-12)
    # Orthogonal states: |⟨0|1⟩|² = 0 => (1 + √(1 - 0)) / 2 = 1
    assert np.isclose(helstrom(ZERO, ONE), 1.0, atol=1e-12)
    # |0⟩ against |+⟩: |⟨0|+⟩|² = 1/2 => (1 + √(1/2)) / 2
    assert np.isclose(helstrom(ZERO, PLUS), (1.0 + np.sqrt(0.5)) / 2.0, atol=1e-12)


def test_swap_test():
    # Phase 15: Swap test
    # |0⟩ with |0⟩ gives 1
    assert np.isclose(swap_test(ZERO, ZERO), 1.0, atol=1e-12)
    # |0⟩ with |1⟩ gives 1/2
    assert np.isclose(swap_test(ZERO, ONE), 0.5, atol=1e-12)

    # Arbitrary pair: circuit output matches (1 + |⟨ψ|φ⟩|²) / 2
    psi = np.array([np.cos(0.4), np.exp(0.7j) * np.sin(0.4)], dtype=complex)
    phi = np.array([np.cos(1.1), np.exp(-0.3j) * np.sin(1.1)], dtype=complex)
    expected = (1.0 + abs(np.vdot(psi, phi)) ** 2) / 2.0
    assert np.isclose(swap_test(psi, phi), expected, atol=1e-12)


def test_grover_once():
    # Phase 16: One Grover iteration on two qubits
    for marked in [0, 1, 2, 3]:
        out = grover_once(marked, qubits=2)
        probs = probabilities(out)
        assert np.isclose(probs[marked], 1.0, atol=1e-12)
        assert np.isclose(abs(np.vdot(basis(marked, 2), out)), 1.0, atol=1e-12)


def test_qft_matrix():
    # Phase 17: Quantum Fourier transform
    for n in [2, 3]:
        dim = 1 << n
        q = qft_matrix(n)
        assert q.shape == (dim, dim)
        # Unitary to 1e-12
        assert np.allclose(q.conj().T @ q, np.eye(dim, dtype=complex), atol=1e-12)
        assert np.allclose(q @ q.conj().T, np.eye(dim, dtype=complex), atol=1e-12)
        # qft_matrix(n) @ basis(0, n) is the uniform ket
        uniform = np.ones(dim, dtype=complex) / np.sqrt(float(dim))
        assert np.allclose(q @ basis(0, n), uniform, atol=1e-12)


def test_cyclic_shift():
    # Phase 18: The clock shift
    t_mat = cyclic_shift(2)
    for k in range(4):
        assert np.allclose(t_mat @ basis(k, 2), basis((k + 1) % 4, 2), atol=1e-12)

    q = qft_matrix(2)
    diag_mat = q.conj().T @ t_mat @ q
    omega = np.exp(2.0j * np.pi / 4.0)
    expected_diag = np.array([omega ** (-k) for k in range(4)], dtype=complex)
    assert np.allclose(expected_diag, [1.0, -1.0j, -1.0, 1.0j], atol=1e-12)
    assert np.allclose(diag_mat, np.diag(expected_diag), atol=1e-12)


def test_chsh_operator():
    # Phase 19: CHSH
    from eigenlab.pauli import X, Z, expectation

    op = chsh_operator()

    # Check matrix construction from A=Z, A'=X, B=(Z+X)/√2, B'=(Z-X)/√2
    A = Z
    A_p = X
    B = (Z + X) / np.sqrt(2.0)
    B_p = (Z - X) / np.sqrt(2.0)
    expected_mat = np.kron(A, B) + np.kron(A, B_p) + np.kron(A_p, B) - np.kron(A_p, B_p)
    assert np.allclose(op.to_matrix(), expected_mat, atol=1e-12)

    # bell("phi+") gives 2√2
    assert np.isclose(expectation(op, bell("phi+")), 2.0 * np.sqrt(2.0), atol=1e-12)

    # basis(0, 2) = |00⟩ gives √2:
    # On |0⟩, ⟨Z⟩ = 1 and ⟨X⟩ = 0, so ⟨A⟩ = 1, ⟨A'⟩ = 0, ⟨B⟩ = 1/√2, ⟨B'⟩ = 1/√2.
    # The four correlators on |00⟩ are:
    #   ⟨A⊗B⟩   = 1 * (1/√2) = 1/√2
    #   ⟨A⊗B'⟩  = 1 * (1/√2) = 1/√2
    #   ⟨A'⊗B⟩  = 0 * (1/√2) = 0
    #   ⟨A'⊗B'⟩ = 0 * (1/√2) = 0
    # The two A' terms vanish and the two A terms contribute 1/√2 each, summing to √2.
    ket_00 = basis(0, 2)
    assert np.isclose(expectation(np.kron(A, B), ket_00), 1.0 / np.sqrt(2.0), atol=1e-12)
    assert np.isclose(expectation(np.kron(A, B_p), ket_00), 1.0 / np.sqrt(2.0), atol=1e-12)
    assert np.isclose(expectation(np.kron(A_p, B), ket_00), 0.0, atol=1e-12)
    assert np.isclose(expectation(np.kron(A_p, B_p), ket_00), 0.0, atol=1e-12)
    assert np.isclose(expectation(op, ket_00), np.sqrt(2.0), atol=1e-12)
