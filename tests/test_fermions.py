import numpy as np

from eigenlab.fermions import (
    jordan_wigner,
    jordan_wigner_a,
    jordan_wigner_adag,
    number_operator,
    total_number_operator,
)
from eigenlab.hamiltonian import to_matrix
from eigenlab.states import basis


def test_jordan_wigner_anticommutation_relations():
    # Phase 14:
    # Map a_p and a†_p to Pauli strings with the frozen qubit order.
    # For n ≤ 4:
    # 1. {a_i, a†_j} = δ_ij I
    # 2. {a_i, a_j} = 0
    # 3. a_i² = 0
    # 4. The vacuum is basis(0, n)
    # 5. a†_p |0⟩ is a single computational-basis vector.

    for n in [1, 2, 3, 4]:
        dim = 1 << n
        identity = np.eye(dim, dtype=complex)
        zero = np.zeros((dim, dim), dtype=complex)
        vac = basis(0, n)

        # Precompute matrix representations for all orbitals
        a_mats = [jordan_wigner_a(p, n).to_matrix() for p in range(n)]
        adag_mats = [jordan_wigner_adag(p, n).to_matrix() for p in range(n)]

        for i in range(n):
            ai = a_mats[i]
            ai_dag = adag_mats[i]

            # 3. a_i² = 0
            assert np.allclose(ai @ ai, zero)
            assert np.allclose(ai_dag @ ai_dag, zero)

            # Check via term operator multiplication as well
            ai_h = jordan_wigner_a(i, n)
            ai_dag_h = jordan_wigner_adag(i, n)
            assert np.allclose((ai_h @ ai_h).to_matrix(), zero)

            # 5. a†_p |0⟩ is a single computational-basis vector:
            # Qubit order contract: qubit 0 is most significant bit.
            # Orbital p occupied => bit p is 1 => index is 1 << (n - 1 - p).
            created_ket = ai_dag @ vac
            expected_index = 1 << (n - 1 - i)
            expected_ket = basis(expected_index, n)
            assert np.allclose(created_ket, expected_ket)

            # Annihilation on vacuum gives 0
            assert np.allclose(ai @ vac, np.zeros(dim, dtype=complex))

            for j in range(n):
                aj = a_mats[j]
                aj_dag = adag_mats[j]

                # 1. {a_i, a†_j} = δ_ij I
                anticom_adag = ai @ aj_dag + aj_dag @ ai
                expected_adag = identity if i == j else zero
                assert np.allclose(anticom_adag, expected_adag)

                # Check via term algebra:
                aj_dag_h = jordan_wigner_adag(j, n)
                term_anticom = (ai_h @ aj_dag_h) + (aj_dag_h @ ai_h)
                assert np.allclose(term_anticom.to_matrix(), expected_adag)

                # 2. {a_i, a_j} = 0
                anticom_aa = ai @ aj + aj @ ai
                assert np.allclose(anticom_aa, zero)


def test_jordan_wigner_number_operators():
    # n_p = a†_p a_p = (I - Z_p)/2
    for n in [1, 2, 3]:
        for p in range(n):
            a_p = jordan_wigner_a(p, n)
            adag_p = jordan_wigner_adag(p, n)
            n_op = number_operator(p, n)
            assert np.allclose((adag_p @ a_p).to_matrix(), n_op.to_matrix())


def test_open_tight_binding_chain():
    # Phase 15:
    # H = −t ∑_i (a†_i a_{i+1} + h.c.) on spinless fermions.
    # Single-particle eigenvalues are −2t cos(π k / (N+1)) for k = 1…N.
    # Number operator ∑ a† a commutes with H, and its expectation on every eigenvector is an integer.
    from eigenlab.fermions import tight_binding_chain
    from eigenlab.hamiltonian import eigensystem
    from eigenlab.pauli import expectation

    for n in [2, 3, 4, 5]:
        t = 1.25
        h_chain = tight_binding_chain(n, t=t)
        h_mat = h_chain.to_matrix()
        n_tot = total_number_operator(n)
        n_mat = n_tot.to_matrix()

        # 1. Number operator commutes with H: [H, N] = 0
        commutator = h_mat @ n_mat - n_mat @ h_mat
        assert np.allclose(commutator, 0.0)

        # 2. Diagonalize H simultaneously with N
        energies, vectors = eigensystem(h_chain, commuting=n_tot)

        # 3. ⟨N⟩ on every eigenvector is an integer
        for col in range(1 << n):
            psi = vectors[:, col]
            exp_n = expectation(n_tot, psi)
            # Must be within 1e-10 of an exact integer in {0, ..., n}
            assert np.isclose(exp_n, round(exp_n), atol=1e-10)
            assert 0 <= round(exp_n) <= n

        # 4. Single-particle sector (N = 1) eigenvalues:
        # ε_k = -2t cos(π k / (n+1)) for k = 1...n
        expected_sp = np.sort([-2.0 * t * np.cos(np.pi * k / (n + 1)) for k in range(1, n + 1)])

        # Collect eigenvalues from the N = 1 eigenspace
        n1_energies = [
            energies[col]
            for col in range(1 << n)
            if round(expectation(n_tot, vectors[:, col])) == 1
        ]
        assert len(n1_energies) == n
        assert np.allclose(np.sort(n1_energies), expected_sp)


def test_hubbard_dimer():
    # Phase 16:
    # Two sites, spin up and spin down, hopping t, on-site U.
    # 4 spin-orbitals: 0↑ (0), 0↓ (1), 1↑ (2), 1↓ (3).
    from eigenlab.fermions import hubbard_dimer, total_spin_z
    from eigenlab.hamiltonian import eigensystem, spectrum
    from eigenlab.pauli import expectation

    n_tot = total_number_operator(4)
    sz_tot = total_spin_z(2)

    # 1. At t = 0, energies are 0 and U with degeneracies counted by occupation
    # In the half-filled N = 2 sector (6 states):
    # - 4 states with 1 electron per site (no double occupancy): energy 0
    # - 2 states with double occupancy on site 0 or site 1: energy U
    u_val = 3.5
    h_t0 = hubbard_dimer(t=0.0, U=u_val)
    _, v_t0 = eigensystem(h_t0, commuting=[n_tot, sz_tot])

    n2_energies_t0 = [
        expectation(h_t0, v_t0[:, col])
        for col in range(16)
        if np.isclose(round(expectation(n_tot, v_t0[:, col])), 2)
    ]
    assert len(n2_energies_t0) == 6
    # Count degeneracies
    zeros_count = sum(np.isclose(e, 0.0) for e in n2_energies_t0)
    u_count = sum(np.isclose(e, u_val) for e in n2_energies_t0)
    assert zeros_count == 4
    assert u_count == 2

    # 2. Reduction by hand to the N = 2, S_z = 0 singlet subspace (3x3):
    # Basis:
    # |S_0⟩ = (|↑, ↓⟩ - |↓, ↑⟩) / √2      (covalent singlet)
    # |D_+⟩ = (|↑↓, 0⟩ + |0, ↑↓⟩) / √2   (symmetric ionic)
    # |D_-⟩ = (|↑↓, 0⟩ - |0, ↑↓⟩) / √2   (antisymmetric ionic)
    # Matrix in this basis:
    # H_3x3 = [[  0, -2t,   0],
    #          [-2t,   U,   0],
    #          [  0,   0,   U]]
    # Eigenvalues of H_3x3 are U and (U ± √(U² + 16 t²))/2.
    # Lowest root: (U - √(U² + 16 t²)) / 2.
    for t in [0.5, 1.0, 2.5]:
        for u in [0.0, 1.0, 4.0]:
            h_dimer = hubbard_dimer(t=t, U=u)
            _, v = eigensystem(h_dimer, commuting=[n_tot, sz_tot])

            # Filter half-filled (N=2), S_z=0 states:
            half_filled_singlet_energies = [
                expectation(h_dimer, v[:, col])
                for col in range(16)
                if np.isclose(round(expectation(n_tot, v[:, col])), 2)
                and np.isclose(expectation(sz_tot, v[:, col]), 0.0, atol=1e-8)
            ]
            # There are 4 states in N=2, Sz=0: the triplet T_0 at 0, and the 3 singlet states
            assert len(half_filled_singlet_energies) == 4

            # Explicit 3x3 matrix from hand calculation
            h_3x3 = np.array(
                [
                    [0.0, -2.0 * t, 0.0],
                    [-2.0 * t, u, 0.0],
                    [0.0, 0.0, u],
                ]
            )
            evals_3x3 = np.sort(np.linalg.eigvalsh(h_3x3))
            lowest_root_algebra = (u - np.sqrt(u**2 + 16.0 * t**2)) / 2.0
            assert np.isclose(evals_3x3[0], lowest_root_algebra)

            # Match lowest eigenvalue in the half-filled subspace
            lowest_in_subspace = min(half_filled_singlet_energies)
            assert np.isclose(lowest_in_subspace, lowest_root_algebra)

    # 3. Check U = 0 (energy -2|t|) and t = 0 (energy 0)
    for t_test in [0.8, 1.5]:
        root_u0 = (0.0 - np.sqrt(0.0**2 + 16.0 * t_test**2)) / 2.0
        assert np.isclose(root_u0, -2.0 * abs(t_test))

    for u_test in [1.2, 5.0]:
        root_t0 = (u_test - np.sqrt(u_test**2 + 0.0)) / 2.0
        assert np.isclose(root_t0, 0.0)


def test_sectors():
    # Phase 17:
    # 1. Tag eigenvectors by ⟨N⟩ and ⟨S_z⟩.
    # 2. Hubbard ground state from phase 16 sits at N = 2, S_z = 0.
    # 3. A one-electron hopping eigenstate sits at N = 1.
    # 4. Reject a vector whose ⟨N⟩ is not within 1e-8 of an integer.
    import pytest
    from eigenlab.fermions import (
        hubbard_dimer,
        jordan_wigner_adag,
        tag_sectors,
        total_number_operator,
        total_spin_z,
    )
    from eigenlab.hamiltonian import eigensystem

    # 2. Hubbard ground state at t=1, U=2 sits at N = 2, S_z = 0
    h_hub = hubbard_dimer(t=1.0, U=2.0)
    n_tot = total_number_operator(4)
    sz_tot = total_spin_z(2)
    energies, vectors = eigensystem(h_hub, commuting=[n_tot, sz_tot])

    ground_state = vectors[:, 0]
    n_tag, sz_tag = tag_sectors(ground_state, n_spatial=2)
    assert n_tag == 2
    assert np.isclose(sz_tag, 0.0)

    # 3. A one-electron hopping eigenstate sits at N = 1
    # Creation of an electron on vacuum:
    vac = basis(0, 4)
    # One electron with spin up (orbital 0) or spin down (orbital 1)
    one_electron_up = (jordan_wigner_adag(0, 4).to_matrix() + jordan_wigner_adag(2, 4).to_matrix()) @ vac
    one_electron_up /= np.linalg.norm(one_electron_up)
    n_1e, sz_1e = tag_sectors(one_electron_up, n_spatial=2)
    assert n_1e == 1
    assert np.isclose(sz_1e, 0.5)

    one_electron_dn = (jordan_wigner_adag(1, 4).to_matrix() - jordan_wigner_adag(3, 4).to_matrix()) @ vac
    one_electron_dn /= np.linalg.norm(one_electron_dn)
    n_1e_dn, sz_1e_dn = tag_sectors(one_electron_dn, n_spatial=2)
    assert n_1e_dn == 1
    assert np.isclose(sz_1e_dn, -0.5)

    # Also check tagging a matrix of all eigenvectors
    all_tags = tag_sectors(vectors, n_spatial=2)
    assert len(all_tags) == 16
    for n_val, sz_val in all_tags:
        assert isinstance(n_val, int)
        assert 0 <= n_val <= 4

    # 4. Reject a vector whose ⟨N⟩ is not within 1e-8 of an integer
    # Create superposition of N=0 and N=1 states:
    superposed_n = (basis(0, 4) + basis(1, 4)) / np.sqrt(2)
    with pytest.raises(ValueError, match="integer"):
        tag_sectors(superposed_n, n_spatial=2)

    # State with ⟨N⟩ = 1.00000002 (violates 1e-8)
    # |ψ⟩ = cos(ε)|1e⟩ + sin(ε)|2e⟩ with sin²(ε) ≈ 2e-8
    eps = 1.5e-4  # sin²(eps) ≈ 2.25e-8 > 1e-8
    bad_state = np.cos(eps) * basis(1, 4) + np.sin(eps) * basis(3, 4)
    with pytest.raises(ValueError, match="integer"):
        tag_sectors(bad_state, n_spatial=2, tol=1e-8)


def test_bravyi_kitaev_operators():
    # Phase 1: Bravyi-Kitaev operators
    from eigenlab.fermions import (
        bravyi_kitaev_a,
        bravyi_kitaev_adag,
        jordan_wigner_a,
        jordan_wigner_adag,
    )
    from eigenlab.pauli import expectation
    from eigenlab.states import basis

    def build_pi(n: int) -> np.ndarray:
        dim = 1 << n
        pi_mat = np.zeros((dim, dim), dtype=complex)
        for x in range(dim):
            bits = [(x >> (n - 1 - q)) & 1 for q in range(n)]
            beta = [0] * n
            for q in range(n):
                i = q + 1
                start = i - (i & -i)
                beta[q] = sum(bits[start:i]) % 2
            y = sum(beta[q] << (n - 1 - q) for q in range(n))
            pi_mat[y, x] = 1.0
        return pi_mat

    # 1. Test oracle Π @ a^{JW} @ Π.T for n <= 4
    for n in [1, 2, 3, 4]:
        pi_mat = build_pi(n)
        for p in range(n):
            bk_mat = bravyi_kitaev_a(p, n).to_matrix()
            jw_mat = jordan_wigner_a(p, n).to_matrix()
            oracle = pi_mat @ jw_mat @ pi_mat.T
            assert np.allclose(bk_mat, oracle)

            bk_dag_mat = bravyi_kitaev_adag(p, n).to_matrix()
            jw_dag_mat = jordan_wigner_adag(p, n).to_matrix()
            oracle_dag = pi_mat @ jw_dag_mat @ pi_mat.T
            assert np.allclose(bk_dag_mat, oracle_dag)

    # 2. Anticommutation relations and dagger for n <= 5
    for n in range(1, 6):
        dim = 1 << n
        ident = np.eye(dim, dtype=complex)
        zero = np.zeros((dim, dim), dtype=complex)
        a_mats = [bravyi_kitaev_a(p, n).to_matrix() for p in range(n)]
        adag_mats = [bravyi_kitaev_adag(p, n).to_matrix() for p in range(n)]

        # Total number operator in BK: N = ∑_p a†_p a_p
        n_bk = np.zeros((dim, dim), dtype=complex)
        for p in range(n):
            n_bk += adag_mats[p] @ a_mats[p]

        vac = basis(0, n)
        for i in range(n):
            ai = a_mats[i]
            ai_dag = adag_mats[i]

            # Dagger check: dagger of bravyi_kitaev_a is bravyi_kitaev_adag
            assert bravyi_kitaev_a(i, n).dagger() == bravyi_kitaev_adag(i, n)
            assert np.allclose(ai.conj().T, ai_dag)

            # a_i² = 0
            assert np.allclose(ai @ ai, zero)
            assert np.allclose(ai_dag @ ai_dag, zero)

            # a†_p applied to vacuum has ⟨N⟩ = 1
            ket_1 = ai_dag @ vac
            norm_sq = np.vdot(ket_1, ket_1).real
            assert np.isclose(norm_sq, 1.0)
            exp_n = np.vdot(ket_1, n_bk @ ket_1).real
            assert np.isclose(exp_n, 1.0)

            for j in range(n):
                aj = a_mats[j]
                aj_dag = adag_mats[j]

                # {a_i, a†_j} = δ_ij I
                anticom_adag = ai @ aj_dag + aj_dag @ ai
                expected_adag = ident if i == j else zero
                assert np.allclose(anticom_adag, expected_adag)

                # {a_i, a_j} = 0
                anticom_aa = ai @ aj + aj @ ai
                assert np.allclose(anticom_aa, zero)

    # 3. For n >= 2, Pauli strings differ from Jordan-Wigner
    for n in [2, 3, 4, 5]:
        differ = False
        for p in range(n):
            jw_terms = jordan_wigner_a(p, n).terms
            bk_terms = bravyi_kitaev_a(p, n).terms
            if set(jw_terms) != set(bk_terms):
                differ = True
                break
        assert differ, f"Bravyi-Kitaev terms must differ from Jordan-Wigner for n={n}"


def test_bravyi_kitaev_same_spectra_tight_binding_and_hubbard():
    # Phase 2: Same spectra, different strings
    from eigenlab.fermions import hubbard_dimer, tight_binding_chain
    from eigenlab.hamiltonian import spectrum

    # 1. Open tight-binding chain at n = 3, 4, 5
    for n in [3, 4, 5]:
        h_jw = tight_binding_chain(n, t=1.25, mapping="jordan_wigner")
        h_bk = tight_binding_chain(n, t=1.25, mapping="bravyi_kitaev")

        evals_jw = spectrum(h_jw)
        evals_bk = spectrum(h_bk)
        assert np.allclose(evals_jw, evals_bk, atol=1e-8)

    # 2. Hubbard dimer at the (t, U) pairs covered by test_hubbard_dimer
    tu_pairs = [
        (0.0, 3.5),
        (0.5, 0.0),
        (0.5, 1.0),
        (0.5, 4.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (1.0, 4.0),
        (2.5, 0.0),
        (2.5, 1.0),
        (2.5, 4.0),
        (0.8, 0.0),
        (1.5, 0.0),
        (0.0, 1.2),
        (0.0, 5.0),
    ]

    for t, u in tu_pairs:
        h_jw = hubbard_dimer(t=t, U=u, mapping="jordan_wigner")
        h_bk = hubbard_dimer(t=t, U=u, mapping="bravyi_kitaev")

        evals_jw = spectrum(h_jw)
        evals_bk = spectrum(h_bk)

        # Full spectra match, including degeneracies
        assert np.allclose(evals_jw, evals_bk, atol=1e-8)


def test_total_spin_squared_fermions():
    # Phase 4: Total spin for fermions
    from eigenlab.fermions import (
        hubbard_dimer,
        total_number_operator,
        total_spin_squared,
        total_spin_z,
    )
    from eigenlab.hamiltonian import eigensystem
    from eigenlab.molecular import h2_sto3g_hamiltonian

    t = 1.0
    n_tot = total_number_operator(4)
    sz_tot = total_spin_z(2)
    s2 = total_spin_squared(2)

    for u in [0.0, 4.0]:
        h_hub = hubbard_dimer(t=t, U=u)
        # Simultaneously diagonalize H, N, and S_z
        evals, evecs = eigensystem(h_hub, commuting=[n_tot, sz_tot, s2])

        # 1. Half-filled singlet: lowest state in N = 2, S_z = 0 sector
        expected_singlet_energy = (u - np.sqrt(u**2 + 16.0 * t**2)) / 2.0
        n2_sz0_states = []
        for col in range(16):
            v = evecs[:, col]
            exp_n = n_tot.expectation(v)
            exp_sz = sz_tot.expectation(v)
            if np.isclose(round(exp_n), 2) and np.isclose(exp_sz, 0.0, atol=1e-8):
                n2_sz0_states.append((evals[col], v))

        n2_sz0_states.sort(key=lambda item: item[0])

        lowest_energy, lowest_vec = n2_sz0_states[0]
        assert np.isclose(lowest_energy, expected_singlet_energy, atol=1e-8)
        assert np.isclose(s2.expectation(lowest_vec), 0.0, atol=1e-8)

        # 2. The N = 2, S_z = 0 triplet has energy 0 and ⟨S²⟩ = 2
        # Filter the triplet state (S^2 ≈ 2) among the N = 2, S_z = 0 states
        triplet_candidates = [
            (e, v)
            for e, v in n2_sz0_states
            if np.isclose(s2.expectation(v), 2.0, atol=1e-6)
        ]
        assert len(triplet_candidates) == 1
        trip_energy, trip_vec = triplet_candidates[0]
        assert np.isclose(trip_energy, 0.0, atol=1e-8)
        assert np.isclose(s2.expectation(trip_vec), 2.0, atol=1e-8)

    # 3. The H₂ ground state at R = 1.401 a.u. has ⟨S²⟩ = 0
    h_h2 = h2_sto3g_hamiltonian(1.401)
    _, evecs_h2 = eigensystem(h_h2)
    gs_h2 = evecs_h2[:, 0]
    s2_4q = total_spin_squared(2)
    assert np.isclose(s2_4q.expectation(gs_h2), 0.0, atol=1e-8)


def test_one_particle_rdm_and_natural_occupations():
    # Phase 5: One-particle density matrix and natural occupations
    from eigenlab.fermions import (
        hubbard_dimer,
        jordan_wigner_adag,
        natural_occupations,
        one_particle_rdm,
        total_number_operator,
    )
    from eigenlab.hamiltonian import eigensystem
    from eigenlab.molecular import h2_sto3g_hamiltonian
    from eigenlab.states import basis

    # 1. Check properties for 2-particle computational states:
    # - γ is Hermitian
    # - Tr(γ) = ⟨N⟩
    # - Every natural occupation lies in [0, 1]
    # - Computational state a†_i a†_j |0⟩ has natural occupations 1, 1, 0, ...
    for n in [3, 4]:
        vac = basis(0, n)
        n_op = total_number_operator(n)
        for i in range(n):
            for j in range(i + 1, n):
                state = (
                    jordan_wigner_adag(i, n).to_matrix()
                    @ jordan_wigner_adag(j, n).to_matrix()
                ) @ vac
                gamma = one_particle_rdm(state, n)

                # γ is Hermitian
                assert np.allclose(gamma, gamma.conj().T)

                # Tr(γ) = ⟨N⟩
                exp_n = n_op.expectation(state)
                assert np.isclose(np.trace(gamma).real, exp_n)

                # Natural occupations
                occ = natural_occupations(gamma)
                # Check natural_occupations(state, n) as well
                occ_direct = natural_occupations(state, n)
                assert np.allclose(occ, occ_direct)

                # Every occupation in [0, 1]
                assert np.all(occ >= -1e-12)
                assert np.all(occ <= 1.0 + 1e-12)

                # Occupations are 1, 1, 0, ...
                expected = np.zeros(n)
                expected[0] = 1.0
                expected[1] = 1.0
                assert np.allclose(occ, expected, atol=1e-10)

    # 2. Hubbard dimer at U = 0, t = 1, full-space ground state has occupations 1, 1, 0, 0
    h_hub = hubbard_dimer(t=1.0, U=0.0)
    _, evecs_hub = eigensystem(h_hub)
    gs_hub = evecs_hub[:, 0]
    occ_hub = natural_occupations(gs_hub, 4)
    assert np.allclose(occ_hub, [1.0, 1.0, 0.0, 0.0], atol=1e-10)

    # 3. Jordan–Wigner H₂ ground state:
    # - Occupations come in two equal pairs and sum to 2
    # - At R = 1.401 a.u., largest occupation is above 0.95
    # - At R = 3.0 a.u., strictly smaller, and still above 1/2
    h_eq = h2_sto3g_hamiltonian(1.401)
    _, evecs_eq = eigensystem(h_eq)
    occ_eq = natural_occupations(evecs_eq[:, 0], 4)

    assert np.isclose(np.sum(occ_eq), 2.0, atol=1e-8)
    assert np.isclose(occ_eq[0], occ_eq[1], atol=1e-8)
    assert np.isclose(occ_eq[2], occ_eq[3], atol=1e-8)
    assert occ_eq[0] > 0.95

    h_str = h2_sto3g_hamiltonian(3.0)
    _, evecs_str = eigensystem(h_str)
    occ_str = natural_occupations(evecs_str[:, 0], 4)

    assert np.isclose(np.sum(occ_str), 2.0, atol=1e-8)
    assert np.isclose(occ_str[0], occ_str[1], atol=1e-8)
    assert np.isclose(occ_str[2], occ_str[3], atol=1e-8)
    assert occ_str[0] < occ_eq[0]
    assert occ_str[0] > 0.5




