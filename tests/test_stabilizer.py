import itertools
import numpy as np
import pytest

from eigenlab.pauli import pauli_comm, pauli_mult
from eigenlab.stabilizer import bits_to_pauli, pauli_to_bits


def test_paulis_as_bit_vectors():
    # Phase 1: Paulis as bit vectors
    # Single qubit check
    assert bits_to_pauli([0], [0]) == "I"
    assert bits_to_pauli([1], [0]) == "X"
    assert bits_to_pauli([0], [1]) == "Z"
    assert bits_to_pauli([1], [1]) == "Y"

    # Round trips are exact
    for length in [1, 2, 3]:
        for p in itertools.product("IXYZ", repeat=length):
            label = "".join(p)
            x, z = pauli_to_bits(label)
            assert x.dtype == np.uint8
            assert z.dtype == np.uint8
            assert bits_to_pauli(x, z) == label

    # Commutation relation: x1·z2 + z1·x2 = 0 (mod 2)
    # Check length-2 labels (256 pairs)
    labels_2 = ["".join(p) for p in itertools.product("IXYZ", repeat=2)]
    for a in labels_2:
        x1, z1 = pauli_to_bits(a)
        for b in labels_2:
            x2, z2 = pauli_to_bits(b)
            symplectic_inner = int((np.dot(x1, z2) + np.dot(z1, x2)) % 2)
            comm_phase, _ = pauli_comm(a, b)
            is_commuting = comm_phase == 0.0
            assert (symplectic_inner == 0) == is_commuting

    # Check length-3 labels (4096 pairs)
    labels_3 = ["".join(p) for p in itertools.product("IXYZ", repeat=3)]
    # All 64 x 64 = 4096 pairs
    for a in labels_3:
        x1, z1 = pauli_to_bits(a)
        for b in labels_3:
            x2, z2 = pauli_to_bits(b)
            symplectic_inner = int((np.dot(x1, z2) + np.dot(z1, x2)) % 2)
            comm_phase, _ = pauli_comm(a, b)
            is_commuting = comm_phase == 0.0
            assert (symplectic_inner == 0) == is_commuting

    # Multiplying two Paulis adds their bit vectors mod 2
    # For every pair of length-2 labels, bits_to_pauli of the sum is the label pauli_mult returns
    for a in labels_2:
        x1, z1 = pauli_to_bits(a)
        for b in labels_2:
            x2, z2 = pauli_to_bits(b)
            sum_x = (x1 + x2) % 2
            sum_z = (z1 + z2) % 2
            mult_label = bits_to_pauli(sum_x, sum_z)
            _, expected_label = pauli_mult(a, b)
            assert mult_label == expected_label


def test_linear_algebra_over_gf2():
    # Phase 2: Linear algebra over GF(2)
    from eigenlab.stabilizer import gf2_nullspace, gf2_rank

    # gf2_rank(I_n) = n
    for n in range(1, 6):
        assert gf2_rank(np.eye(n, dtype=np.uint8)) == n

    # The all-ones 3x3 matrix has rank 1
    all_ones_3 = np.ones((3, 3), dtype=np.uint8)
    assert gf2_rank(all_ones_3) == 1
    ns_ones = gf2_nullspace(all_ones_3)
    assert len(ns_ones) == 2
    for v in ns_ones:
        assert np.all((all_ones_3 @ v) % 2 == 0)
    assert gf2_rank(ns_ones) == 2

    # The Hamming [7, 4] parity-check matrix, with rows 0001111, 0110011, 1010101
    h_hamming = np.array(
        [
            [0, 0, 0, 1, 1, 1, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [1, 0, 1, 0, 1, 0, 1],
        ],
        dtype=np.uint8,
    )
    assert gf2_rank(h_hamming) == 3
    # Real rank is also 3
    assert np.linalg.matrix_rank(h_hamming.astype(float)) == 3

    null_hamming = gf2_nullspace(h_hamming)
    assert len(null_hamming) == 4
    for v in null_hamming:
        assert np.all((h_hamming @ v) % 2 == 0)
    assert gf2_rank(null_hamming) == 4

    # Case where GF(2) rank and real rank differ:
    # Rows 110, 011, 101 have GF(2) rank 2 (sum to zero mod 2) but real rank 3
    m_diff = np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
        ],
        dtype=np.uint8,
    )
    assert gf2_rank(m_diff) == 2
    assert np.linalg.matrix_rank(m_diff.astype(float)) == 3
    ns_diff = gf2_nullspace(m_diff)
    assert len(ns_diff) == 1
    assert np.all((m_diff @ ns_diff[0]) % 2 == 0)
    assert gf2_rank(ns_diff) == 1


def test_z2_symmetries():
    # Phase 3: The Z2 symmetries of a Hamiltonian
    # Derivation:
    # A Z-type Pauli string has bit representation (x=0, z=a).
    # A Hamiltonian term has bit representation (x=x_t, z=z_t).
    # The symplectic inner product between them is:
    #   x_1 · z_2 + z_1 · x_2 (mod 2) = 0 · z_t + a · x_t (mod 2) = a · x_t (mod 2).
    # Therefore, the Z-type Pauli string commutes with the term if and only if
    #   a · x_t = 0 (mod 2).
    # For the string to commute with every term in H, a must satisfy M_X a = 0 (mod 2),
    # where M_X is the matrix whose rows are the x-parts x_t of each term.
    # Hence, the independent Z2 symmetries are the basis of the GF(2) nullspace of M_X.
    from eigenlab.codes import stabilizer_group
    from eigenlab.molecular import h2_sto3g_hamiltonian
    from eigenlab.pauli import pauli
    from eigenlab.stabilizer import z2_symmetries

    h2 = h2_sto3g_hamiltonian(1.401)
    gens = z2_symmetries(h2)

    # Jordan–Wigner H2 at R = 1.401 has exactly 3 generators
    # (Bravyi, Gambetta, Mezzacapo and Temme, arXiv:1701.08213 taper H2 to one qubit)
    assert len(gens) == 3
    assert "IIII" not in gens

    # Every returned label consists only of I and Z
    for g in gens:
        assert set(g).issubset({"I", "Z"})

    # Every returned label commutes with every term (pauli_comm) and with H.to_matrix() to 1e-12
    h_mat = h2.to_matrix()
    for g in gens:
        g_mat = pauli(g)
        # Matrix commutator [G, H] = 0
        comm_mat = g_mat @ h_mat - h_mat @ g_mat
        assert np.allclose(comm_mat, 0.0, atol=1e-12)

        for coeff, term_lbl in h2.terms:
            if abs(coeff) > 1e-12:
                comm_phase, _ = pauli_comm(g, term_lbl)
                assert comm_phase == 0.0

    # The group they generate (stabilizer_group) contains ZIZI, IZIZ and ZZII.
    # These are:
    # - ZIZI: parity of N_↑ = n0 + n2
    # - IZIZ: parity of N_↓ = n1 + n3
    # - ZZII: parity of the σ_g occupation n0 + n1, which the double excitation σ_g² ↔ σ_u² changes by 2.
    group = stabilizer_group(gens)
    assert "ZIZI" in group
    assert "IZIZ" in group
    assert "ZZII" in group


def test_z2_taper_one_clifford_per_symmetry():
    # Phase 4: One Clifford per symmetry
    from eigenlab.hamiltonian import Hamiltonian, z2_taper
    from eigenlab.molecular import h2_sto3g_hamiltonian
    from eigenlab.pauli import pauli, pauli_mult
    from eigenlab.stabilizer import _gf2_rref, bits_to_pauli, pauli_to_bits, z2_symmetries
    from eigenlab.states import ONE, PLUS, ZERO

    MINUS = (ZERO - ONE) / np.sqrt(2.0)

    h2 = h2_sto3g_hamiltonian(1.401)
    gens = z2_symmetries(h2)
    assert len(gens) == 3

    z_rows = [pauli_to_bits(g)[1] for g in gens]
    rref, pivot_cols = _gf2_rref(np.array(z_rows))
    taus = [bits_to_pauli(np.zeros(4), row) for row in rref]
    assert len(pivot_cols) == 3

    dim = 16
    U = np.eye(dim, dtype=complex)
    u_list = []
    for tau, q in zip(taus, pivot_cols):
        x_lbl = ["I"] * 4
        x_lbl[q] = "X"
        x_str = "".join(x_lbl)
        Ui = (pauli(x_str) + pauli(tau)) / np.sqrt(2.0)
        # Ui is Hermitian and unitary
        assert np.allclose(Ui @ Ui.conj().T, np.eye(dim))
        assert np.allclose(Ui @ pauli(tau) @ Ui, pauli(x_str))
        u_list.append(Ui)
        U = Ui @ U

    # 1. U is unitary
    assert np.allclose(U @ U.conj().T, np.eye(dim))

    # 2. U τ_i U† = X_{q_i}
    for tau, q in zip(taus, pivot_cols):
        x_lbl = ["I"] * 4
        x_lbl[q] = "X"
        x_str = "".join(x_lbl)
        assert np.allclose(U @ pauli(tau) @ U.conj().T, pauli(x_str), atol=1e-12)

    # 3. U H U† has only I and X on every pivot qubit
    h_mat = h2.to_matrix()
    h_prime = U @ h_mat @ U.conj().T

    # Track term-list conjugation by each U_i
    current_h = h2
    for tau, q in zip(taus, pivot_cols):
        x_lbl = ["I"] * 4
        x_lbl[q] = "X"
        x_str = "".join(x_lbl)
        new_terms = []
        for coeff, lbl in current_h.terms:
            if lbl[q] in ("I", "X"):
                new_terms.append((coeff, lbl))
            else:
                ph1, p1 = pauli_mult(x_str, lbl)
                ph2, p2 = pauli_mult(p1, tau)
                new_terms.append((coeff * ph1 * ph2, p2))
        current_h = Hamiltonian(new_terms).simplify()

    assert np.allclose(current_h.to_matrix(), h_prime, atol=1e-12)
    for _, lbl in current_h.terms:
        for q in pivot_cols:
            assert lbl[q] in ("I", "X")

    # 4. The term-list result of z2_taper matches the matrix result for every sign choice
    for signs in itertools.product([1, -1], repeat=3):
        h_tapered = z2_taper(h2, signs=signs, generators=gens)
        mat_tapered = h_tapered.to_matrix()

        # Matrix projection onto the sector defined by signs
        # Remaining qubit is qubit 3 (dim = 2)
        v_cols = []
        for b in [ZERO, ONE]:
            s_states = [PLUS if s == 1 else MINUS for s in signs]
            full_ket = np.kron(s_states[0], np.kron(s_states[1], np.kron(s_states[2], b)))
            v_cols.append(full_ket)
        Vs = np.column_stack(v_cols)
        expected_mat = Vs.conj().T @ h_prime @ Vs
        assert np.allclose(mat_tapered, expected_mat, atol=1e-12)


def test_h2_on_one_qubit():
    # Phase 5: H2 on one qubit
    from eigenlab.hamiltonian import spectrum, taper, z2_taper
    from eigenlab.molecular import h2_sto3g_hamiltonian
    from eigenlab.stabilizer import z2_symmetries

    for R in [1.0, 1.401, 2.0, 3.0]:
        h_jw = h2_sto3g_hamiltonian(R, mapping="jordan_wigner")
        spec_full = spectrum(h_jw)
        gens = z2_symmetries(h_jw)
        assert len(gens) == 3

        # The eight one-qubit tapered spectra, over all sign choices
        tapered_specs = []
        for signs in itertools.product([1, -1], repeat=3):
            h1 = z2_taper(h_jw, signs=signs, generators=gens)
            assert h1.qubits == 1
            sp1 = spectrum(h1)
            assert len(sp1) == 2
            tapered_specs.append(sp1)

        # Together equal the sixteen-level four-qubit spectrum to 1e-10, multiplicities included
        combined = np.sort(np.concatenate(tapered_specs))
        assert np.allclose(combined, spec_full, atol=1e-10)

        # The lowest tapered level is the four-qubit ground energy to 1e-10
        lowest = min(min(sp) for sp in tapered_specs)
        assert np.isclose(lowest, spec_full[0], atol=1e-10)

        # At R = 1.401 it is the cited −1.13727 to 1e-5
        if R == 1.401:
            assert np.isclose(lowest, -1.13727, atol=1e-5)

        # Equals the ground energy of the fifth haul's two-qubit parity taper to 1e-10
        h_parity = h2_sto3g_hamiltonian(R, mapping="parity")
        h_par_taper = taper(h_parity, [1, 3], [+1, +1])
        assert np.isclose(lowest, spectrum(h_par_taper)[0], atol=1e-10)




