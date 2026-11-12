import numpy as np
import pytest

from eigenlab.clifford import one_qubit_cliffords
from eigenlab.pauli import X, Y, Z


def test_twenty_four_elements():
    # Phase 13: Twenty-four elements
    # The one-qubit Clifford group modulo global phase is isomorphic to the octahedral group O ~= S_4,
    # having order 24.
    # The map C ↦ (C X C†, C Z C†) is injective because:
    # 1. X can be mapped to any of the 6 signed single-qubit Paulis: {+X, -X, +Y, -Y, +Z, -Z}.
    # 2. Since {X, Z} = 0, C Z C† must anticommute with C X C†.
    #    Out of the 6 signed Paulis, exactly 2 commute with C X C† (itself and its negative),
    #    leaving exactly 4 signed Paulis that anticommute with C X C†.
    # Thus there are 6 * 4 = 24 possible pairs (C X C†, C Z C†), and because the action of C
    # on X and Z completely determines its action on Y = i X Z and on the entire matrix modulo global phase,
    # the mapping is injective and bijection holds.
    cliffords = one_qubit_cliffords()

    # 1. There are 24 elements
    assert len(cliffords) == 24

    signed_paulis = {
        "+X": X,
        "-X": -X,
        "+Y": Y,
        "-Y": -Y,
        "+Z": Z,
        "-Z": -Z,
    }

    def identify_signed_pauli(mat: np.ndarray) -> str:
        for name, p in signed_paulis.items():
            if np.allclose(mat, p, atol=1e-8):
                return name
        raise AssertionError(f"Matrix {mat} is not a signed Pauli")

    pairs = set()
    for c in cliffords:
        # Phase check: first nonzero entry is real and positive
        for val in c.ravel():
            if abs(val) > 1e-10:
                assert val.real > 0
                assert np.isclose(val.imag, 0.0, atol=1e-10)
                break

        # 2. Each conjugates X, Y and Z to signed Paulis
        cx = identify_signed_pauli(c @ X @ c.conj().T)
        cy = identify_signed_pauli(c @ Y @ c.conj().T)
        cz = identify_signed_pauli(c @ Z @ c.conj().T)

        # 3. Pair (C X C†, C Z C†)
        pairs.add((cx, cz))

    # The map C ↦ (C X C†, C Z C†) is injective: all 24 pairs are distinct
    assert len(pairs) == 24


def test_unitary_2_design():
    # Phase 14: A unitary 2-design
    # The second frame potential of a group G is (1/|G|) ∑_{C ∈ G} |Tr C|⁴.
    # Haar measure gives 2 for d >= 2 (Gross, Audenaert and Eisert, J. Math. Phys. 48, 052104 (2007)).
    # Count by class for the 24 one-qubit Cliffords:
    # - the identity gives 16 (1 element with |Tr| = 2)
    # - the 3 Paulis are traceless (|Tr| = 0, giving 0)
    # - the 8 order-3 rotations have |Tr| = 1, giving 8 (8 * 1⁴ = 8)
    # - the 6 quarter turns have |Tr| = √2, giving 24 (6 * (√2)⁴ = 24)
    # - the 6 edge half-turns are traceless (|Tr| = 0, giving 0)
    # Total sum is 16 + 0 + 8 + 24 + 0 = 48, so the frame potential is 48/24 = 2.
    from eigenlab.pauli import I, X, Y, Z
    from eigenlab.states import ONE, PLUS, PLUS_I, ZERO

    cliffords = one_qubit_cliffords()
    abs_traces = [abs(np.trace(c)) for c in cliffords]

    count_id = sum(np.isclose(t, 2.0) for t in abs_traces)
    count_traceless = sum(np.isclose(t, 0.0) for t in abs_traces)
    count_rot3 = sum(np.isclose(t, 1.0) for t in abs_traces)
    count_qturn = sum(np.isclose(t, np.sqrt(2.0)) for t in abs_traces)

    assert count_id == 1
    assert count_traceless == 9  # 3 Paulis + 6 edge half-turns
    assert count_rot3 == 8
    assert count_qturn == 6

    # Contributions:
    # Identity: 1 * 16 = 16
    # Order-3: 8 * 1 = 8
    # Quarter-turns: 6 * 4 = 24
    # Traceless: 9 * 0 = 0
    # Total sum: 16 + 8 + 24 = 48 -> 48 / 24 = 2
    sum_traces_4 = sum(t**4 for t in abs_traces)
    assert np.isclose(sum_traces_4, 48.0, atol=1e-12)
    fp_clifford = sum_traces_4 / len(cliffords)
    assert np.isclose(fp_clifford, 2.0, atol=1e-12)

    # The one-qubit Pauli group, modulo phase, gives 4 and is not a 2-design
    pauli_group = [I, X, Y, Z]
    fp_pauli = sum(abs(np.trace(p)) ** 4 for p in pauli_group) / len(pauli_group)
    assert np.isclose(fp_pauli, 4.0, atol=1e-12)

    # The six stabilizer states are a state 2-design:
    # (1/36) ∑_{ij} |⟨ψ_i|ψ_j⟩|⁴ = 1/3 = 2/(d(d+1))
    # The pairs contribute 6 identical pairs at 1, 6 antipodal pairs at 0, and 24 pairs at 1/4.
    minus = (ZERO - ONE) / np.sqrt(2.0)
    minus_i = (ZERO - 1j * ONE) / np.sqrt(2.0)
    stabilizer_states = [ZERO, ONE, PLUS, minus, PLUS_I, minus_i]

    overlaps_4 = []
    for si in stabilizer_states:
        for sj in stabilizer_states:
            overlaps_4.append(abs(np.vdot(si, sj)) ** 4)

    assert len(overlaps_4) == 36
    count_identical = sum(np.isclose(o, 1.0) for o in overlaps_4)
    count_antipodal = sum(np.isclose(o, 0.0) for o in overlaps_4)
    count_quarter = sum(np.isclose(o, 0.25) for o in overlaps_4)

    assert count_identical == 6
    assert count_antipodal == 6
    assert count_quarter == 24

    state_fp = sum(overlaps_4) / len(overlaps_4)
    d = 2
    expected_haar = 2.0 / (d * (d + 1))  # 1/3
    assert np.isclose(state_fp, 1.0 / 3.0, atol=1e-12)
    assert np.isclose(state_fp, expected_haar, atol=1e-12)


def test_twirling():
    # Phase 15: Twirling
    from eigenlab.channels import amplitude_damping, depolarize
    from eigenlab.clifford import twirl
    from eigenlab.states import ONE, PLUS, PLUS_I, ZERO, density

    minus = (ZERO - ONE) / np.sqrt(2.0)
    minus_i = (ZERO - 1j * ONE) / np.sqrt(2.0)
    stabilizer_states = [ZERO, ONE, PLUS, minus, PLUS_I, minus_i]

    for p in [0.0, 0.3, 1.0]:
        ch = lambda rho: amplitude_damping(rho, p)
        tw = twirl(ch)

        # Entanglement fidelity: F_e = ∑_k |Tr E_k|²/d² = (1 + √(1−p))²/4
        fe = (1.0 + np.sqrt(1.0 - p)) ** 2 / 4.0
        lam = (4.0 * fe - 1.0) / 3.0
        depol_p = 3.0 * (1.0 - lam) / 4.0

        # Twirl is depolarizing: ρ ↦ λρ + (1−λ) I/2 = depolarize(ρ, 3(1−λ)/4)
        for s in stabilizer_states:
            rho = density(s)
            res_tw = tw(rho)
            res_depol = depolarize(rho, depol_p)
            assert np.allclose(res_tw, res_depol, atol=1e-12)

        # Average fidelity over the six stabilizer states is (2F_e + 1)/3 = (1 + λ)/2
        fids = [np.real(np.vdot(s, tw(density(s)) @ s)) for s in stabilizer_states]
        avg_fid = np.mean(fids)
        expected_avg_fid = (2.0 * fe + 1.0) / 3.0
        assert np.isclose(avg_fid, expected_avg_fid, atol=1e-12)
        assert np.isclose(avg_fid, (1.0 + lam) / 2.0, atol=1e-12)

        # At p = 1, every state goes to |0⟩, and the average of |⟨ψ|0⟩|² over the six states is 1/2
        if p == 1.0:
            overlaps_p1 = [abs(np.vdot(s, ZERO)) ** 2 for s in stabilizer_states]
            assert np.isclose(np.mean(overlaps_p1), 0.5, atol=1e-12)


def test_randomized_benchmarking_exact_average():
    # Phase 16: Benchmarking, averaged exactly
    # Derivation:
    # A standard randomized benchmarking sequence of length m applies m Clifford gates C_1, ..., C_m,
    # each interleaved with a noise channel Λ, followed by the noiseless inversion gate C_{inv} = (C_m ... C_1)†.
    # Defining prefix unitaries D_i = C_i ... C_1 (with D_0 = I), the noise operator at step i,
    # conjugated back into the reference frame, becomes:
    #   D_i† Λ D_i.
    # Because each C_i is independently and uniformly distributed over the Clifford group,
    # each prefix D_i is also independent and uniformly distributed over the Clifford group.
    # Therefore, the expectation over the sequence factors into the independent Clifford twirls
    # of each noise channel:
    #   E[D_i† Λ D_i] = Twirl(Λ).
    # In Phase 15 we verified that the twirl of amplitude damping over the Clifford 2-design is a depolarizing channel:
    #   Twirl(Λ)(ρ) = λ ρ + (1 - λ) I/2,
    # with parameter λ = (4 F_e - 1)/3, where F_e = (1 + √(1 - p))² / 4 is the entanglement fidelity.
    # Composing m independent identical depolarizing channels gives:
    #   Twirl(Λ)^m(ρ) = λ^m ρ + (1 - λ^m) I/2.
    # Starting from the state ρ_0 = |0⟩⟨0| and measuring the survival probability of |0⟩ gives:
    #   F(m) = ⟨0| Twirl(Λ)^m(|0⟩⟨0|) |0⟩
    #        = λ^m ⟨0|0⟩⟨0|0⟩ + (1 - λ^m) ⟨0| I/2 |0⟩
    #        = λ^m + (1 - λ^m)/2 = (1 + λ^m)/2.
    import itertools
    from eigenlab.channels import amplitude_damping
    from eigenlab.clifford import one_qubit_cliffords
    from eigenlab.states import ZERO, density

    cliffords = one_qubit_cliffords()
    assert len(cliffords) == 24
    rho0 = density(ZERO)

    for p in [0.1, 0.4]:
        fe = (1.0 + np.sqrt(1.0 - p)) ** 2 / 4.0
        lam = (4.0 * fe - 1.0) / 3.0
        noise = lambda rho: amplitude_damping(rho, p)

        for m in [1, 2, 3]:
            total_f = 0.0
            count = 0
            for seq in itertools.product(cliffords, repeat=m):
                count += 1
                rho = rho0
                u_net = np.eye(2, dtype=complex)
                for c in seq:
                    rho = noise(c @ rho @ c.conj().T)
                    u_net = c @ u_net
                rho_final = u_net.conj().T @ rho @ u_net
                total_f += rho_final[0, 0].real

            avg_f = total_f / count
            expected_f = (1.0 + lam**m) / 2.0
            assert count == 24**m
            assert np.isclose(avg_f, expected_f, atol=1e-12)


def test_t_is_not_clifford():
    # Phase 17: T is not Clifford
    # Algebra using |0⟩⟨1| = (X + iY)/2:
    #   X = |0⟩⟨1| + |1⟩⟨0|
    #   T |0⟩⟨1| T† = |0⟩⟨1| e^{-iπ/4} = (cos(π/4) - i sin(π/4)) |0⟩⟨1| = (1 - i)/√2 |0⟩⟨1|
    #   T |1⟩⟨0| T† = e^{iπ/4} |1⟩⟨0| = (1 + i)/√2 |1⟩⟨0|
    # Since |0⟩⟨1| = (X + iY)/2 and |1⟩⟨0| = (X - iY)/2:
    #   T X T† = (1 - i)/√2 (X + iY)/2 + (1 + i)/√2 (X - iY)/2
    #          = [ (1 - i)(X + iY) + (1 + i)(X - iY) ] / (2√2)
    #          = [ (X + iY - iX + Y) + (X - iY + iX + Y) ] / (2√2)
    #          = (2X + 2Y) / (2√2) = (X + Y)/√2.
    # Because (X + Y)/√2 has support on both X and Y with coefficient 1/√2,
    # it is not a signed Pauli operator (which must be ±X, ±Y, or ±Z).
    #
    # Furthermore, (X + Y)/√2 is itself a Clifford unitary:
    #   ((X + Y)/√2) X ((X + Y)/√2)† = Y
    #   ((X + Y)/√2) Y ((X + Y)/√2)† = X
    #   ((X + Y)/√2) Z ((X + Y)/√2)† = -Z
    # Thus (X + Y)/√2 sends X → Y, Y → X, Z → −Z, which are all signed Paulis.
    # Consequently, T conjugates X to a Clifford unitary, and since T Z T† = Z is Pauli,
    # and T Y T† = T (-i Z X) T† = -i Z (X + Y)/√2 = (Y - X)/√2 is also Clifford,
    # T conjugates every Pauli into the Clifford group, which places T in the third level
    # of the Clifford hierarchy (C_3 \ C_2).
    from eigenlab.clifford import T, canonicalize_phase, one_qubit_cliffords
    from eigenlab.pauli import X, Y, Z

    # 1. T = diag(1, e^{iπ/4})
    expected_T = np.diag([1.0, np.exp(1j * np.pi / 4.0)])
    assert np.allclose(T, expected_T, atol=1e-12)

    # Projector algebra check
    ket0_bra1 = (X + 1j * Y) / 2.0
    ket1_bra0 = (X - 1j * Y) / 2.0
    assert np.allclose(ket0_bra1, [[0, 1], [0, 0]])
    assert np.allclose(ket1_bra0, [[0, 0], [1, 0]])
    assert np.allclose(X, ket0_bra1 + ket1_bra0)

    t_01 = T @ ket0_bra1 @ T.conj().T
    t_10 = T @ ket1_bra0 @ T.conj().T
    assert np.allclose(t_01, np.exp(-1j * np.pi / 4.0) * ket0_bra1)
    assert np.allclose(t_10, np.exp(1j * np.pi / 4.0) * ket1_bra0)

    # 2. T X T† = (X + Y)/√2, not a signed Pauli
    txt = T @ X @ T.conj().T
    expected_txt = (X + Y) / np.sqrt(2.0)
    assert np.allclose(txt, expected_txt, atol=1e-12)

    signed_paulis = [X, -X, Y, -Y, Z, -Z]
    assert not any(np.allclose(txt, p, atol=1e-6) for p in signed_paulis)

    # 3. (X + Y)/√2 is itself a Clifford sending X → Y, Y → X, Z → −Z
    u = expected_txt
    assert np.allclose(u @ X @ u.conj().T, Y, atol=1e-12)
    assert np.allclose(u @ Y @ u.conj().T, X, atol=1e-12)
    assert np.allclose(u @ Z @ u.conj().T, -Z, atol=1e-12)

    cliffords = one_qubit_cliffords()
    u_canon = canonicalize_phase(u)
    assert any(np.allclose(u_canon, c, atol=1e-8) for c in cliffords)

    # 4. T conjugates every Pauli into the Clifford group
    for p_mat in [X, Y, Z]:
        conj_p = canonicalize_phase(T @ p_mat @ T.conj().T)
        assert any(np.allclose(conj_p, c, atol=1e-8) for c in cliffords)


def test_stabilizer_renyi_entropy():
    # Phase 18: Stabilizer Rényi entropy
    from eigenlab.clifford import T, one_qubit_cliffords, stabilizer_renyi_entropy
    from eigenlab.pauli import expectation_label
    from eigenlab.states import ONE, PLUS, PLUS_I, ZERO, basis

    # 1. It is 0 on the six one-qubit stabilizer states
    minus = (ZERO - ONE) / np.sqrt(2.0)
    minus_i = (ZERO - 1j * ONE) / np.sqrt(2.0)
    for s in [ZERO, ONE, PLUS, minus, PLUS_I, minus_i]:
        assert np.isclose(stabilizer_renyi_entropy(s), 0.0, atol=1e-12)

    # 2. It is 0 on the state vectors of the phase-7 circuits with n <= 4
    circuits_n_le_4 = [
        (2, [("h", 0), ("cnot", 0, 1), ("s", 1)]),
        (3, [("h", 0), ("cnot", 0, 1), ("cnot", 1, 2), ("s", 0), ("h", 2)]),
        (
            4,
            [
                ("h", 0),
                ("h", 1),
                ("h", 2),
                ("h", 3),
                ("cnot", 0, 1),
                ("cnot", 1, 2),
                ("cnot", 2, 3),
                ("s", 1),
                ("s", 3),
            ],
        ),
        (
            4,
            [
                ("h", 0),
                ("s", 0),
                ("cnot", 0, 2),
                ("h", 1),
                ("cnot", 1, 3),
                ("s", 2),
                ("cnot", 2, 3),
                ("h", 3),
            ],
        ),
    ]

    H_mat = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0)
    S_mat = np.array([[1, 0], [0, 1j]], dtype=complex)
    X_mat = np.array([[0, 1], [1, 0]], dtype=complex)
    P0 = np.array([[1, 0], [0, 0]], dtype=complex)
    P1 = np.array([[0, 0], [0, 1]], dtype=complex)

    for n, gates in circuits_n_le_4:
        state = basis(0, n)
        for g in gates:
            if g[0] == "h":
                op = 1
                for i in range(n):
                    u = H_mat if i == g[1] else np.eye(2, dtype=complex)
                    op = np.kron(op, u) if i > 0 else u
                state = op @ state
            elif g[0] == "s":
                op = 1
                for i in range(n):
                    u = S_mat if i == g[1] else np.eye(2, dtype=complex)
                    op = np.kron(op, u) if i > 0 else u
                state = op @ state
            elif g[0] == "cnot":
                c, t = g[1], g[2]
                m0 = 1
                m1 = 1
                for i in range(n):
                    op0 = P0 if i == c else np.eye(2, dtype=complex)
                    op1 = P1 if i == c else (X_mat if i == t else np.eye(2, dtype=complex))
                    m0 = np.kron(m0, op0) if i > 0 else op0
                    m1 = np.kron(m1, op1) if i > 0 else op1
                state = (m0 + m1) @ state
        assert np.isclose(stabilizer_renyi_entropy(state), 0.0, atol=1e-12)

    # 3. |T⟩ = T|+⟩ has ⟨X⟩ = ⟨Y⟩ = 1/√2 and ⟨Z⟩ = 0, so the sum is 1 + 1/4 + 1/4 = 3/2 and M₂ = log₂(4/3)
    t_ket = T @ PLUS
    assert np.isclose(expectation_label(t_ket, "X"), 1.0 / np.sqrt(2.0), atol=1e-12)
    assert np.isclose(expectation_label(t_ket, "Y"), 1.0 / np.sqrt(2.0), atol=1e-12)
    assert np.isclose(expectation_label(t_ket, "Z"), 0.0, atol=1e-12)

    expected_m2_t = np.log2(4.0 / 3.0)
    assert np.isclose(stabilizer_renyi_entropy(t_ket), expected_m2_t, atol=1e-12)

    # 4. |T⟩^{⊗k} has k log₂(4/3) for k = 1 … 4
    prod_state = t_ket
    for k in range(1, 5):
        assert np.isclose(stabilizer_renyi_entropy(prod_state), k * expected_m2_t, atol=1e-10)
        prod_state = np.kron(prod_state, t_ket)

    # 5. Each of the 24 one-qubit Cliffords leaves M₂ unchanged on a fixed non-stabilizer state
    cliffords = one_qubit_cliffords()
    for c in cliffords:
        c_state = c @ t_ket
        assert np.isclose(stabilizer_renyi_entropy(c_state), expected_m2_t, atol=1e-12)





