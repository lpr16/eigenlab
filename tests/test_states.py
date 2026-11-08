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
    relative_entropy,
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


def test_relative_entropy():
    # Phase 20: Relative entropy
    rho_0 = density(ZERO)
    i_over_2 = 0.5 * np.eye(2, dtype=complex)

    assert np.isclose(relative_entropy(rho_0, i_over_2), 1.0, atol=1e-12)
    assert relative_entropy(i_over_2, i_over_2) == 0.0

    # sigma must be positive definite; otherwise raise ValueError
    with pytest.raises(ValueError, match="positive definite"):
        relative_entropy(i_over_2, rho_0)


def test_bipartite_entropy_at_any_cut():
    # Phase 35: Entropy at any cut
    # At the middle cut it equals entanglement_entropy. ghz(n) has 1 bit at every cut.
    # A product state has 0.
    from eigenlab.states import bipartite_entropy

    for n in [2, 3, 4, 5, 6]:
        g = ghz(n)
        assert np.isclose(bipartite_entropy(g, n // 2), entanglement_entropy(g), atol=1e-12)
        for cut in range(1, n):
            assert np.isclose(bipartite_entropy(g, cut), 1.0, atol=1e-12)

        prod = basis(0, n)
        for cut in range(1, n):
            assert bipartite_entropy(prod, cut) == 0.0

    # Non-uniform state at middle cut matches entanglement_entropy
    psi_4 = 0.6 * basis(0b0000, 4) + 0.5 * basis(0b0110, 4) + np.sqrt(1.0 - 0.36 - 0.25) * basis(0b1111, 4)
    assert np.isclose(bipartite_entropy(psi_4, 2), entanglement_entropy(psi_4), atol=1e-12)


def test_teleportation():
    # Phase 18: Teleportation
    from eigenlab.states import teleport_outcomes

    generic_ket = np.cos(0.3) * ZERO + np.exp(0.7j) * np.sin(0.3) * ONE
    test_kets = [ZERO, ONE, PLUS, PLUS_I, generic_ket]

    for psi in test_kets:
        # 1) Pure-state trajectory using post_measure
        outcomes = teleport_outcomes(psi)
        assert len(outcomes) == 4
        for (m0, m1), (prob, ket_corr) in outcomes.items():
            assert np.isclose(prob, 0.25, atol=1e-12)
            # Corrected qubit 2 equals |psi> up to a global phase
            overlap = abs(np.vdot(psi, ket_corr))
            assert np.isclose(overlap, 1.0, atol=1e-12)
            assert np.allclose(density(ket_corr), density(psi), atol=1e-12)

        # 2) Density-matrix trajectory using projectors and partial_trace
        outcomes_rho = teleport_outcomes(density(psi))
        for (m0, m1), (prob, rho_corr) in outcomes_rho.items():
            assert np.isclose(prob, 0.25, atol=1e-12)
            assert np.allclose(rho_corr, density(psi), atol=1e-12)


def test_teleportation_through_werner_pair():
    # Phase 19: Teleportation through a Werner pair
    from eigenlab.channels import depolarize
    from eigenlab.states import teleport_average, werner_state

    psi = np.cos(0.3) * ZERO + np.exp(0.7j) * np.sin(0.3) * ONE
    rho = density(psi)
    i2 = np.eye(2, dtype=complex)

    for p in (0.2, 0.5, 0.9):
        rho_w = werner_state(p)
        out = teleport_average(psi, resource=rho_w)
        expected = p * rho + (1.0 - p) * 0.5 * i2
        assert np.allclose(out, expected, atol=1e-12)
        assert np.allclose(out, depolarize(rho, 0.75 * (1.0 - p)), atol=1e-12)
        fid = float(np.real(np.vdot(psi, out @ psi)))
        assert np.isclose(fid, 0.5 * (1.0 + p), atol=1e-12)

    # Classical bound 2/3 beaten iff p > 1/3 iff concurrence(rho_W(p)) == max(0, (3p - 1)/2) > 0
    for p in (0.3, 0.4):
        rho_w = werner_state(p)
        out = teleport_average(psi, resource=rho_w)
        fid = float(np.real(np.vdot(psi, out @ psi)))
        conc = concurrence(rho_w)
        if p == 0.3:
            assert fid < 2.0 / 3.0
            assert np.isclose(conc, 0.0, atol=1e-12)
        else:
            assert fid > 2.0 / 3.0
            assert np.isclose(conc, (3.0 * p - 1.0) / 2.0, atol=1e-12)
            assert conc > 0.0


def test_superdense_coding():
    # Phase 20: Superdense coding
    from eigenlab.pauli import I, X, Z
    from eigenlab.states import superdense_decode, superdense_encode

    expected_bells = {
        (0, 0): bell("phi+"),
        (0, 1): bell("psi+"),
        (1, 0): bell("phi-"),
        (1, 1): bell("psi-"),
    }
    encoded_states = []
    for (b0, b1), ref_bell in expected_bells.items():
        enc = superdense_encode(b0, b1)
        assert np.allclose(enc, ref_bell, atol=1e-12)
        encoded_states.append(enc)

        dec = superdense_decode(enc)
        idx = (b0 << 1) | b1
        probs = probabilities(dec)
        assert np.isclose(probs[idx], 1.0, atol=1e-12)
        assert np.allclose(density(dec), density(basis(idx, 2)), atol=1e-12)

        # In circuit order (Z^{b0} then X^{b1}, i.e. operator X^{b1} Z^{b0}), (1, 1) produces -|11>
        op_circuit = (X if b1 else I) @ (Z if b0 else I)
        dec_circuit = superdense_decode(np.kron(op_circuit, I) @ bell("phi+"))
        expected_ket = -basis(0b11, 2) if (b0, b1) == (1, 1) else basis(idx, 2)
        assert np.allclose(dec_circuit, expected_ket, atol=1e-12)

    # The four encoded Bell states are mutually orthogonal
    for i in range(4):
        for j in range(4):
            expected_ov = 1.0 if i == j else 0.0
            assert np.isclose(abs(np.vdot(encoded_states[i], encoded_states[j])), expected_ov, atol=1e-12)


def test_phase_estimation():
    # Phase 21: Phase estimation
    from eigenlab.states import phase_estimation

    def exact_qpe_probs(phi: float, m: int) -> np.ndarray:
        dim = 1 << m
        out = np.zeros(dim, dtype=float)
        for j in range(dim):
            delta = phi - j / float(dim)
            if np.isclose(np.mod(delta, 1.0), 0.0, atol=1e-14) or np.isclose(np.mod(delta, 1.0), 1.0, atol=1e-14):
                out[j] = 1.0
            else:
                out[j] = (np.sin(np.pi * dim * delta) ** 2) / ((dim**2) * (np.sin(np.pi * delta) ** 2))
        return out

    # phi = 5/8 with m = 3 returns j = 5 with probability 1
    probs_5_8, pre_5_8 = phase_estimation(5.0 / 8.0, 3, return_pre_iqft=True)
    x_3 = np.arange(8, dtype=float)
    expected_pre_5_8 = np.exp(2.0j * np.pi * (5.0 / 8.0) * x_3) / np.sqrt(8.0)
    assert np.allclose(pre_5_8, expected_pre_5_8, atol=1e-12)
    assert np.isclose(probs_5_8[5], 1.0, atol=1e-12)
    assert np.allclose(probs_5_8, exact_qpe_probs(5.0 / 8.0, 3), atol=1e-12)

    # phi = 1/3 with m = 4, 6 and phi = 0.2137 with m = 7 match formula to 1e-12
    for phi, m in [(1.0 / 3.0, 4), (1.0 / 3.0, 6), (0.2137, 7)]:
        probs, pre = phase_estimation(phi, m, return_pre_iqft=True)
        dim = 1 << m
        x = np.arange(dim, dtype=float)
        expected_pre = np.exp(2.0j * np.pi * phi * x) / np.sqrt(float(dim))
        assert np.allclose(pre, expected_pre, atol=1e-12)
        assert np.allclose(probs, exact_qpe_probs(phi, m), atol=1e-12)
        # Most likely outcome has probability at least 4 / pi^2
        assert np.max(probs) >= 4.0 / (np.pi**2)


def test_mutual_information_and_conditional_entropy():
    # Phase 22: Mutual information and conditional entropy
    from eigenlab.states import conditional_entropy, mutual_information

    def binary_entropy(p: float) -> float:
        return float(-p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p))

    # Bell pair: I = 2 and S(A|B) = -1
    phi_plus = bell("phi+")
    assert np.isclose(mutual_information(phi_plus, 2, [0], [1]), 2.0, atol=1e-12)
    assert np.isclose(conditional_entropy(phi_plus, 2, [0], [1]), -1.0, atol=1e-12)

    # |0> ⊗ |+>: I = 0
    prod = np.kron(ZERO, PLUS)
    assert np.isclose(mutual_information(prod, 2, [0], [1]), 0.0, atol=1e-12)

    # ghz(3), qubits 0 and 1: I = 1 and S(A|B) = 0
    g3 = ghz(3)
    assert np.isclose(mutual_information(g3, 3, [0], [1]), 1.0, atol=1e-12)
    assert np.isclose(conditional_entropy(g3, 3, [0], [1]), 0.0, atol=1e-12)

    # w_state(3), qubits 0 and 1: rho_AB and rho_A both have eigenvalues 2/3, 1/3, so I = h(1/3)
    w3 = w_state(3)
    rho_ab = partial_trace(w3, 3, keep=[0, 1])
    rho_a = partial_trace(w3, 3, keep=[0])
    evals_ab = np.sort(np.real(np.linalg.eigvalsh(rho_ab)))[::-1]
    evals_a = np.sort(np.real(np.linalg.eigvalsh(rho_a)))[::-1]
    assert np.allclose(evals_ab[:2], [2.0 / 3.0, 1.0 / 3.0], atol=1e-12)
    assert np.allclose(evals_ab[2:], 0.0, atol=1e-12)
    assert np.allclose(evals_a, [2.0 / 3.0, 1.0 / 3.0], atol=1e-12)
    assert np.isclose(mutual_information(w3, 3, [0], [1]), binary_entropy(1.0 / 3.0), atol=1e-12)


def test_holevo_quantity():
    # Phase 23: Holevo quantity
    from eigenlab.states import holevo

    def binary_entropy(p: float) -> float:
        return float(-p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p))

    def classical_entropy(probs) -> float:
        p = np.asarray(probs, dtype=float)
        p = p[p > 1e-15]
        return float(-np.sum(p * np.log2(p)))

    # |0> and |+> with equal weight: average state has eigenvalues (1 +- 1/sqrt(2))/2
    avg_0_plus = 0.5 * density(ZERO) + 0.5 * density(PLUS)
    evals = np.sort(np.real(np.linalg.eigvalsh(avg_0_plus)))
    lam_plus = 0.5 * (1.0 + 1.0 / np.sqrt(2.0))
    lam_minus = 0.5 * (1.0 - 1.0 / np.sqrt(2.0))
    assert np.allclose(evals, [lam_minus, lam_plus], atol=1e-12)

    chi_0_plus = holevo([0.5, 0.5], [ZERO, PLUS])
    assert np.isclose(chi_0_plus, binary_entropy(lam_plus), atol=1e-12)
    assert chi_0_plus <= classical_entropy([0.5, 0.5]) + 1e-12

    # Two orthogonal states with equal weight: 1 bit
    chi_orth = holevo([0.5, 0.5], [ZERO, ONE])
    assert np.isclose(chi_orth, 1.0, atol=1e-12)
    assert chi_orth <= classical_entropy([0.5, 0.5]) + 1e-12

    # Two copies of the same state: 0
    chi_same = holevo([0.5, 0.5], [PLUS, PLUS])
    assert chi_same == 0.0
    assert chi_same <= classical_entropy([0.5, 0.5]) + 1e-12

    # Mixed-state ensemble check: chi <= H(p)
    rho_mix = 0.7 * density(ZERO) + 0.3 * density(ONE)
    chi_mix = holevo([0.3, 0.7], [rho_mix, PLUS_I])
    assert 0.0 <= chi_mix <= classical_entropy([0.3, 0.7]) + 1e-12


def test_qfi_pure_state():
    # Phase 24: Quantum Fisher information of a pure state
    from eigenlab.hamiltonian import Hamiltonian
    from eigenlab.states import qfi

    for n in range(1, 9):
        gen = Hamiltonian([(0.5, "I" * i + "Z" + "I" * (n - 1 - i)) for i in range(n)])
        dim = 1 << n

        # |+>^{⊗n}: n
        plus_n = np.ones(dim, dtype=complex) / np.sqrt(float(dim))
        assert np.isclose(qfi(plus_n, gen), float(n), atol=1e-12)

        # basis(0, n): 0
        assert qfi(basis(0, n), gen) == 0.0

        # ghz(n): n^2 (for n = 1, (|0> + |1>)/sqrt(2) = |+> has 1^2 = 1)
        g_n = plus_n if n == 1 else ghz(n)
        assert np.isclose(qfi(g_n, gen), float(n * n), atol=1e-12)








