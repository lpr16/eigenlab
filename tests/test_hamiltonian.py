import numpy as np

from eigenlab.hamiltonian import (
    Hamiltonian,
    eigensystem,
    heisenberg_dimer,
    heisenberg_dimer_spin_blocks,
    heisenberg_dimer_terms,
    parity,
    parity_blocks,
    spectrum,
    term_expectation,
    to_matrix,
    transverse_ising,
    transverse_ising_terms,
    xxz_dimer,
    xxz_dimer_terms,
)
from eigenlab.pauli import expectation
from eigenlab.states import basis, bell


def test_heisenberg_dimer_singlet_and_triplet():
    energies = spectrum(heisenberg_dimer())
    assert np.allclose(energies, [-3.0, 1.0, 1.0, 1.0])


def test_transverse_ising_is_hermitian_and_ordered():
    energies = spectrum(transverse_ising(3, coupling=1.0, field=0.5))
    assert energies.shape == (8,)
    assert np.all(np.diff(energies) >= -1e-12)


def test_hamiltonians_as_term_lists():
    # Phase 6: to_matrix and term-list expectation reproduce heisenberg_dimer,
    # transverse_ising, and expectation.
    # 1. Heisenberg dimer
    h_terms = heisenberg_dimer_terms()
    h_mat = heisenberg_dimer()
    assert np.allclose(h_terms.to_matrix(), h_mat)
    assert np.allclose(to_matrix(h_terms.terms), h_mat)

    # Test term expectation vs matrix expectation on Bell states
    for name in ["phi+", "phi-", "psi+", "psi-"]:
        state = bell(name)
        exp_matrix = expectation(h_mat, state)
        exp_term = h_terms.expectation(state)
        exp_func = term_expectation(h_terms, state)
        exp_pauli = expectation(h_terms, state)
        assert np.isclose(exp_term, exp_matrix)
        assert np.isclose(exp_func, exp_matrix)
        assert np.isclose(exp_pauli, exp_matrix)

    # 2. Transverse Ising model (n = 3)
    ti_terms = transverse_ising_terms(3, coupling=1.2, field=0.7)
    ti_mat = transverse_ising(3, coupling=1.2, field=0.7)
    assert np.allclose(ti_terms.to_matrix(), ti_mat)
    assert np.allclose(to_matrix(ti_terms.terms), ti_mat)

    # Check expectation on computational basis states |000> ... |111>
    for idx in range(8):
        b = basis(idx, 3)
        assert np.isclose(ti_terms.expectation(b), expectation(ti_mat, b))
        assert np.isclose(term_expectation(ti_terms.terms, b), expectation(ti_mat, b))

    # 3. Check spectrum accepts Hamiltonian directly
    assert np.allclose(spectrum(ti_terms), spectrum(ti_mat))


def test_eigenvectors():
    # Phase 7:
    # 1. Return ascending energies and orthonormal eigenvectors
    h = heisenberg_dimer()
    energies, vectors = eigensystem(h)
    assert energies.shape == (4,)
    assert vectors.shape == (4, 4)
    # Ascending order
    assert np.all(np.diff(energies) >= 0)
    # Orthonormal eigenvectors: V† V = I
    assert np.allclose(vectors.conj().T @ vectors, np.eye(4))
    # H v_k = E_k v_k
    for k in range(4):
        assert np.allclose(h @ vectors[:, k], energies[k] * vectors[:, k])

    # 2. Ground state of XX + YY + ZZ is the singlet (|01⟩ - |10⟩)/√2 up to global phase
    ground_state = vectors[:, 0]
    singlet = bell("psi-")  # (|01⟩ - |10⟩)/√2
    overlap = abs(np.vdot(singlet, ground_state))
    assert np.isclose(overlap, 1.0)

    # 3. The triplet is a subspace: projector onto three states at energy +1 has rank 3
    triplet_vectors = vectors[:, 1:4]
    p_triplet = triplet_vectors @ triplet_vectors.conj().T
    # Projector properties: P² = P, P† = P
    assert np.allclose(p_triplet @ p_triplet, p_triplet)
    assert np.allclose(p_triplet.conj().T, p_triplet)
    # Rank is trace of projector
    rank = int(np.round(np.trace(p_triplet).real))
    assert rank == 3

    # 4. spectrum stays the eigenvalues alone
    assert np.allclose(spectrum(h), energies)
    assert np.allclose(spectrum(heisenberg_dimer_terms()), energies)


def test_xxz_dimer():
    # Phase 8: H = XX + YY + Δ ZZ
    # Algebra:
    # (XX + YY)|00⟩ = 0, Δ ZZ|00⟩ = Δ |00⟩ => E(|00⟩) = Δ
    # (XX + YY)|11⟩ = 0, Δ ZZ|11⟩ = Δ |11⟩ => E(|11⟩) = Δ
    # (XX + YY)|ψ+⟩ = 2|ψ+⟩, Δ ZZ|ψ+⟩ = -Δ |ψ+⟩ => E(|ψ+⟩) = 2 - Δ
    # (XX + YY)|ψ-⟩ = -2|ψ-⟩, Δ ZZ|ψ-⟩ = -Δ |ψ-⟩ => E(|ψ-⟩) = -2 - Δ
    state_00 = basis(0, 2)
    state_11 = basis(3, 2)
    psi_plus = bell("psi+")
    singlet = bell("psi-")

    for delta in [0.0, 0.5, 1.0, 2.5, -1.0]:
        h = xxz_dimer(delta)
        # Check action on all 4 states
        assert np.isclose(expectation(h, state_00), delta)
        assert np.isclose(expectation(h, state_11), delta)
        assert np.isclose(expectation(h, psi_plus), 2.0 - delta)
        assert np.isclose(expectation(h, singlet), -2.0 - delta)

        # Eigenvalues must match the set {Δ, Δ, 2 - Δ, -2 - Δ}
        expected_energies = np.sort([delta, delta, 2.0 - delta, -2.0 - delta])
        assert np.allclose(spectrum(h), expected_energies)
        assert np.allclose(spectrum(xxz_dimer_terms(delta)), expected_energies)

    # Specific checks from specification:
    # Δ = 1 matches Phase 7 (energies: -3, 1, 1, 1)
    assert np.allclose(spectrum(xxz_dimer(1.0)), [-3.0, 1.0, 1.0, 1.0])
    assert np.allclose(xxz_dimer(1.0), heisenberg_dimer())

    # Δ = 0 has energies −2, 0, 0, 2
    assert np.allclose(spectrum(xxz_dimer(0.0)), [-2.0, 0.0, 0.0, 2.0])


def test_transverse_ising_hand_limits():
    # Phase 9:
    # 1. For two sites and zero field, H = −J ZZ has energies −J, −J, +J, +J
    # Proof: ZZ|00⟩ = |00⟩, ZZ|11⟩ = |11⟩ (E = -J)
    #        ZZ|01⟩ = -|01⟩, ZZ|10⟩ = -|10⟩ (E = +J)
    for j_val in [1.0, 2.5, 0.7]:
        h_two_site_zero_field = transverse_ising(2, coupling=j_val, field=0.0)
        energies = spectrum(h_two_site_zero_field)
        assert np.allclose(energies, [-j_val, -j_val, j_val, j_val])

    # 2. Observables for n = 3, J = 1 at h = 0 and h = 20
    # Observables ∑ X and ∑ Z
    sum_x = Hamiltonian([(1.0, "XII"), (1.0, "IXI"), (1.0, "IIX")])
    sum_z = Hamiltonian([(1.0, "ZII"), (1.0, "IZI"), (1.0, "IIZ")])

    # At h = 0 (ferromagnetic limit):
    # Ground space is spanned by |000⟩ and |111⟩.
    # ⟨∑ X⟩ = 0 identically. ⟨∑ Z⟩ = ±3.
    _, v_h0 = eigensystem(transverse_ising(3, coupling=1.0, field=0.0))
    gs_h0 = v_h0[:, 0]
    exp_x_h0 = sum_x.expectation(gs_h0)
    exp_z_h0 = sum_z.expectation(gs_h0)
    assert np.isclose(exp_x_h0, 0.0, atol=1e-12)
    assert np.isclose(abs(exp_z_h0), 3.0, atol=1e-12)

    # At large field h = 20:
    # Ground state approaches |+⟩^⊗3.
    # ⟨∑ X⟩ → n = 3, ⟨∑ Z⟩ → 0.
    _, v_h20 = eigensystem(transverse_ising(3, coupling=1.0, field=20.0))
    gs_h20 = v_h20[:, 0]
    exp_x_h20 = sum_x.expectation(gs_h20)
    exp_z_h20 = sum_z.expectation(gs_h20)
    # Exact perturbation: ⟨∑ X⟩ = 3 - 3/(4 h²) + O(h⁻⁴) ≈ 3 - 3/1600 = 2.998125...
    # Here h=20, coupling=1 => ⟨∑ X⟩ > 2.998, ⟨∑ Z⟩ < 1e-12
    assert 2.998 < exp_x_h20 <= 3.0
    assert np.isclose(exp_z_h20, 0.0, atol=1e-12)


def test_bogoliubov_spectrum_transverse_ising():
    # Phase 3: Bogoliubov spectrum of the open transverse Ising chain
    import itertools

    # One-site sanity check, written next to the assertion:
    # H = −h Z has energies −h and +h, and the formula with A = [2h], B = 0, ε = 2h reproduces them.
    h_1site = 1.75
    A_1 = np.array([[2.0 * h_1site]], dtype=float)
    B_1 = np.array([[0.0]], dtype=float)
    bdg_1 = np.block([[A_1, B_1], [-B_1, -A_1]])
    evals_1 = np.sort(np.real(np.linalg.eigvals(bdg_1)))
    eps_1 = evals_1[1:]  # ε = 2h
    assert np.isclose(eps_1[0], 2.0 * h_1site)
    energies_1site = np.sort([(nu - 0.5) * eps_1[0] for nu in [0, 1]])
    assert np.allclose(energies_1site, [-h_1site, +h_1site])

    # Multi-site checks: The test builds this Bogoliubov matrix itself.
    # It does not call spectrum on the spin operator and rename the result.
    parameter_pairs = [(1.0, 0.0), (1.0, 0.5), (1.0, 1.0), (0.7, 1.3)]
    for n in [2, 3, 4, 5]:
        for J, h in parameter_pairs:
            A = np.zeros((n, n), dtype=float)
            B = np.zeros((n, n), dtype=float)
            for i in range(n):
                A[i, i] = 2.0 * h
            for i in range(n - 1):
                A[i, i + 1] = -J
                A[i + 1, i] = -J
                B[i, i + 1] = -J
                B[i + 1, i] = +J

            # 2n × 2n Bogoliubov–de Gennes matrix [[ A, B ], [ −B*, −A* ]]
            bdg = np.block([[A, B], [-B, -A]])

            evals = np.sort(np.real(np.linalg.eigvals(bdg)))
            # Take the n algebraically largest (nonnegative quasiparticle energies ε_k)
            eps = evals[n:]

            # Every many-body energy is E(ν) = ∑_k (ν_k − 1/2) ε_k
            many_body_energies = [
                sum((nu_k - 0.5) * eps[k] for k, nu_k in enumerate(nu))
                for nu in itertools.product([0, 1], repeat=n)
            ]
            e_bogoliubov = np.sort(many_body_energies)

            # Compare against the full spin spectrum, matching to 1e-8 with degeneracies included
            e_spin = spectrum(transverse_ising(n, coupling=J, field=h))
            assert np.allclose(e_bogoliubov, e_spin, atol=1e-8)


def test_total_spin_qubits():
    # Phase 4: Total spin for qubits
    from eigenlab.hamiltonian import total_spin, xxz_dimer_terms
    from eigenlab.pauli import expectation
    from eigenlab.states import basis, bell

    s2 = total_spin(2)

    # Singlet (|01⟩ − |10⟩) / √2 has ⟨S²⟩ = 0
    singlet = bell("psi-")
    assert np.isclose(s2.expectation(singlet), 0.0, atol=1e-12)

    # Each of the three triplet states has ⟨S²⟩ = 2
    triplet_m1 = basis(3, 2)  # |11⟩, S_z = -1
    triplet_0 = bell("psi+")  # (|01⟩ + |10⟩) / √2, S_z = 0
    triplet_p1 = basis(0, 2)  # |00⟩, S_z = +1

    for trip in [triplet_m1, triplet_0, triplet_p1]:
        assert np.isclose(s2.expectation(trip), 2.0, atol=1e-12)

    # Verification against S² = 3/2 + (XX + YY + ZZ) / 2
    # Singlet of XX + YY + ZZ is at −3 -> S² = 1.5 + (-3)/2 = 0
    # Triplet of XX + YY + ZZ is at +1 -> S² = 1.5 + 1/2 = 2
    h_heis = xxz_dimer_terms(1.0)
    exp_singlet_heis = h_heis.expectation(singlet)
    assert np.isclose(exp_singlet_heis, -3.0)
    assert np.isclose(1.5 + 0.5 * exp_singlet_heis, 0.0)

    for trip in [triplet_m1, triplet_0, triplet_p1]:
        exp_trip_heis = h_heis.expectation(trip)
        assert np.isclose(exp_trip_heis, 1.0)
        assert np.isclose(1.5 + 0.5 * exp_trip_heis, 2.0)


def test_parity_transverse_ising():
    # Phase 21: Parity of the transverse Ising chain
    for n in [2, 3, 4]:
        p_op = parity(n)
        assert p_op.terms == [(1.0 + 0.0j, "X" * n)]
        p_mat = p_op.to_matrix()
        h_mat = transverse_ising(n, coupling=1.0, field=1.0)
        assert np.allclose(p_mat @ h_mat - h_mat @ p_mat, 0.0, atol=1e-12)

        # Even cat (|0...0⟩ + |1...1⟩)/√2 has ⟨P⟩ = +1; odd cat has ⟨P⟩ = -1
        even_cat = (basis(0, n) + basis((1 << n) - 1, n)) / np.sqrt(2.0)
        odd_cat = (basis(0, n) - basis((1 << n) - 1, n)) / np.sqrt(2.0)
        assert np.isclose(p_op.expectation(even_cat), 1.0, atol=1e-12)
        assert np.isclose(p_op.expectation(odd_cat), -1.0, atol=1e-12)

    # Ground state at n = 3, J = 1, h = 1 has ⟨P⟩ = +1
    _, evecs_3 = eigensystem(transverse_ising(3, coupling=1.0, field=1.0))
    gs_3 = evecs_3[:, 0]
    assert np.isclose(parity(3).expectation(gs_3), 1.0, atol=1e-12)


def test_parity_blocks(monkeypatch):
    # Phase 22: Parity blocks
    import eigenlab.hamiltonian

    def _forbidden_simultaneous(*args, **kwargs):
        raise RuntimeError("simultaneous_eigensystem must not be called")

    monkeypatch.setattr(eigenlab.hamiltonian, "simultaneous_eigensystem", _forbidden_simultaneous)

    param_pairs = [(1.0, 0.0), (1.0, 0.5), (1.0, 1.0), (0.7, 1.3)]
    for n in [2, 3, 4]:
        for j_val, h_val in param_pairs:
            h_mat = transverse_ising(n, coupling=j_val, field=h_val)
            spec_even, spec_odd = parity_blocks(h_mat, n)
            assert spec_even.shape == (1 << (n - 1),)
            assert spec_odd.shape == (1 << (n - 1),)
            union_spec = np.sort(np.concatenate([spec_even, spec_odd]))
            full_spec = spectrum(h_mat)
            assert np.allclose(union_spec, full_spec, atol=1e-8)


def test_heisenberg_dimer_spin_blocks():
    # Phase 24: Spin blocks of the Heisenberg dimer
    sz = Hamiltonian([(0.5, "ZI"), (0.5, "IZ")])
    assert np.isclose(sz.expectation(basis(0, 2)), 1.0)
    assert np.isclose(sz.expectation(basis(1, 2)), 0.0)
    assert np.isclose(sz.expectation(basis(2, 2)), 0.0)
    assert np.isclose(sz.expectation(basis(3, 2)), -1.0)

    blocks = heisenberg_dimer_spin_blocks()
    # S_z = 0 block on |01⟩ and |10⟩ has energies -3 and +1
    evals_sz0 = np.sort(np.real(np.linalg.eigvalsh(blocks[0])))
    assert np.allclose(evals_sz0, [-3.0, 1.0], atol=1e-12)

    # |00⟩ (S_z = +1) and |11⟩ (S_z = -1) each have energy +1
    assert np.allclose(np.real(np.linalg.eigvalsh(blocks[1])), [1.0], atol=1e-12)
    assert np.allclose(np.real(np.linalg.eigvalsh(blocks[-1])), [1.0], atol=1e-12)




def test_lanczos_full_krylov_space():
    # Phase 48: Lanczos, exact when the Krylov space is full
    # The Krylov space of a start vector is spanned by its components on the eigenvectors of H
    # with distinct eigenvalues. When it is invariant, T is H restricted to it and its
    # eigenvalues are exact.
    from eigenlab.hamiltonian import lanczos

    # |00⟩ is an eigenvector of the Heisenberg dimer at +1: H|00⟩ = ZZ|00⟩ = |00⟩,
    # since (XX + YY)|00⟩ = |11⟩ − |11⟩ = 0. The first residual vanishes.
    h = heisenberg_dimer()
    start = basis(0, 2)
    values, t_mat = lanczos(h, start, 4)
    assert t_mat.shape == (1, 1)
    assert np.isclose(t_mat[0, 0], expectation(h, start), atol=1e-12)
    assert np.allclose(values, [1.0], atol=1e-12)

    # transverse_ising(3, 1, 1) has eight distinct eigenvalues, and (|000⟩ + |001⟩)/√2 overlaps
    # every eigenvector, so eight steps span the whole space.
    h = transverse_ising(3, coupling=1.0, field=1.0)
    start = (basis(0, 3) + basis(1, 3)) / np.sqrt(2.0)
    exact, vectors = eigensystem(h)
    assert np.all(np.diff(exact) > 1e-6)
    assert np.all(np.abs(vectors.conj().T @ start) > 1e-3)

    values, t_mat = lanczos(h, start, 8)
    assert t_mat.shape == (8, 8)
    assert np.isclose(t_mat[0, 0], expectation(h, start), atol=1e-12)
    assert np.allclose(values, exact, atol=1e-8)


def test_two_site_ising_modes():
    # Phase 6: Two-site modes
    # For n = 2, H = -J Z_0 Z_1 - h (X_0 + X_1) has eigenvalues ±√(J² + 4h²) and ±J.
    # In terms of Bogoliubov quasiparticle modes 0 <= ε_1 <= ε_2:
    #   E(0, 0) = -(ε_1 + ε_2) / 2 = -√(J² + 4h²)
    #   E(1, 0) =  (ε_1 - ε_2) / 2 = -J
    # Adding and subtracting gives:
    #   ε_1 = √(J² + 4h²) - J
    #   ε_2 = √(J² + 4h²) + J
    from eigenlab.hamiltonian import transverse_ising_bdg_matrix

    for j_val, h_val in [(1.0, 0.5), (0.7, 1.3), (1.0, 1.0)]:
        rad = np.sqrt(j_val**2 + 4.0 * h_val**2)
        expected_mb = np.sort([-rad, -j_val, j_val, rad])
        assert np.allclose(spectrum(transverse_ising(2, j_val, h_val)), expected_mb, atol=1e-12)

        eps_1 = rad - j_val
        eps_2 = rad + j_val
        bdg = transverse_ising_bdg_matrix(2, j_val, h_val)
        bdg_evals = np.sort(np.real(np.linalg.eigvalsh(bdg)))
        nonneg = bdg_evals[2:]
        assert np.allclose(nonneg, [eps_1, eps_2], atol=1e-12)

    # At h = 0, ε_1 = 0 (a zero mode) and the ground state at -J is doubly degenerate
    bdg_h0 = transverse_ising_bdg_matrix(2, 1.0, 0.0)
    nonneg_h0 = np.sort(np.real(np.linalg.eigvalsh(bdg_h0)))[2:]
    assert np.isclose(nonneg_h0[0], 0.0, atol=1e-12)
    assert np.isclose(nonneg_h0[1], 2.0, atol=1e-12)
    spec_h0 = spectrum(transverse_ising(2, 1.0, 0.0))
    assert np.allclose(spec_h0[:2], [-1.0, -1.0], atol=1e-12)


def test_critical_ising_chain():
    # Phase 7: The critical chain
    from eigenlab.hamiltonian import transverse_ising_bdg_matrix

    scaled_gaps = []
    for n in [2, 3, 4, 5, 6, 7, 8, 16, 32, 64]:
        bdg = transverse_ising_bdg_matrix(n, coupling=1.0, field=1.0)
        bdg_evals = np.sort(np.real(np.linalg.eigvalsh(bdg)))
        nonneg = bdg_evals[n:]

        m = np.arange(1, n + 1, dtype=float)
        expected_eps = 4.0 * np.sin((2.0 * m - 1.0) * np.pi / (2.0 * (2.0 * n + 1.0)))
        assert np.allclose(nonneg, expected_eps, atol=1e-12)

        eps_1 = nonneg[0]
        # Since sin(x) < x for x > 0: (2n + 1) * eps_1 < 2 * pi
        scaled = (2 * n + 1) * eps_1
        assert scaled < 2.0 * np.pi

        if n <= 8:
            mb_spec = spectrum(transverse_ising(n, coupling=1.0, field=1.0))
            assert np.isclose(mb_spec[1] - mb_spec[0], eps_1, atol=1e-12)

        if n in (8, 16, 32, 64):
            scaled_gaps.append(scaled)

    # Strictly increasing over n = 8, 16, 32, 64
    assert all(scaled_gaps[i] < scaled_gaps[i + 1] for i in range(len(scaled_gaps) - 1))


def test_translation_operator():
    # Phase 8: The translation operator
    from eigenlab.hamiltonian import translation
    from eigenlab.pauli import pauli

    # On three qubits, T basis(4, 3) = basis(2, 3): |100⟩ -> |010⟩
    t3 = translation(3)
    assert np.allclose(t3 @ basis(4, 3), basis(2, 3), atol=1e-12)

    for n in [3, 4, 5]:
        dim = 1 << n
        ident = np.eye(dim, dtype=complex)
        t_mat = translation(n)

        # T is unitary and T^n = I
        assert np.allclose(t_mat @ t_mat.conj().T, ident, atol=1e-12)
        assert np.allclose(np.linalg.matrix_power(t_mat, n), ident, atol=1e-12)

        # T Z_i T† = Z_{i+1 mod n} for every i
        for i in range(n):
            zi = pauli("I" * i + "Z" + "I" * (n - 1 - i))
            z_next = pauli("I" * ((i + 1) % n) + "Z" + "I" * (n - 1 - ((i + 1) % n)))
            assert np.allclose(t_mat @ zi @ t_mat.conj().T, z_next, atol=1e-12)


def test_heisenberg_ring():
    # Phase 9: Heisenberg ring
    import pytest

    from eigenlab.hamiltonian import heisenberg_ring, total_spin, translation

    # Reject n = 2
    with pytest.raises(ValueError):
        heisenberg_ring(2)

    for n in [4, 5, 6, 7, 8]:
        h_op = heisenberg_ring(n)
        h_mat = h_op.to_matrix()
        t_mat = translation(n)
        s2_mat = total_spin(n).to_matrix()
        sum_z = Hamiltonian([(1.0, "I" * i + "Z" + "I" * (n - 1 - i)) for i in range(n)]).to_matrix()

        assert np.allclose(h_mat @ t_mat - t_mat @ h_mat, 0.0, atol=1e-12)
        assert np.allclose(h_mat @ s2_mat - s2_mat @ h_mat, 0.0, atol=1e-12)
        assert np.allclose(h_mat @ sum_z - sum_z @ h_mat, 0.0, atol=1e-12)

        # basis(0, n) has energy n
        assert np.isclose(h_op.expectation(basis(0, n)), float(n), atol=1e-12)


def test_momentum_blocks():
    # Phase 10: Momentum blocks
    from eigenlab.hamiltonian import heisenberg_ring, momentum_blocks

    for n in [4, 5, 6, 7, 8]:
        h = heisenberg_ring(n)
        blks = momentum_blocks(h, n)
        assert len(blks) == n
        assert sum(len(b) for b in blks) == (1 << n)
        union_spec = np.sort(np.concatenate(blks))
        assert np.allclose(union_spec, spectrum(h), atol=1e-10)


def test_heisenberg_ring_one_magnon():
    # Phase 11: One magnon
    from eigenlab.hamiltonian import heisenberg_ring, total_spin

    for n in [4, 5, 6, 7, 8]:
        h_op = heisenberg_ring(n)
        h_mat = h_op.to_matrix()
        s2_op = total_spin(n)

        single_indices = [1 << (n - 1 - j) for j in range(n)]
        h_sub = h_mat[np.ix_(single_indices, single_indices)]
        evals_sub = np.sort(np.real(np.linalg.eigvalsh(h_sub)))

        ks = [2.0 * np.pi * m / n for m in range(n)]
        expected = np.sort([n - 4.0 * (1.0 - np.cos(k)) for k in ks])
        assert np.allclose(evals_sub, expected, atol=1e-12)

        for m, k in enumerate(ks):
            magnon = sum(
                np.exp(1j * k * j) * basis(1 << (n - 1 - j), n) for j in range(n)
            ) / np.sqrt(n)
            e_k = n - 4.0 * (1.0 - np.cos(k))
            assert np.allclose(h_mat @ magnon, e_k * magnon, atol=1e-12)
            s2_val = s2_op.expectation(magnon)
            if m == 0:
                assert np.isclose(s2_val, (n / 2.0) * (n / 2.0 + 1.0), atol=1e-12)
            else:
                assert np.isclose(s2_val, (n / 2.0 - 1.0) * (n / 2.0), atol=1e-12)


def test_heisenberg_ring_four_sites_sublattice():
    # Phase 12: Four sites from two sublattice spins
    # On 4 sites, each even site {0, 2} couples to both odd sites {1, 3}:
    # H = 4 S_A · S_B = 2 [S(S+1) - S_A(S_A+1) - S_B(S_B+1)]
    from eigenlab.hamiltonian import heisenberg_ring

    levels: list[float] = []
    for s_a in (0, 1):
        for s_b in (0, 1):
            for s in range(abs(s_a - s_b), s_a + s_b + 1):
                energy = 2.0 * (s * (s + 1) - s_a * (s_a + 1) - s_b * (s_b + 1))
                multiplicity = 2 * s + 1
                levels.extend([energy] * multiplicity)

    expected = np.sort(levels)
    assert len(expected) == 16
    assert np.allclose(
        expected,
        [-8.0] * 1 + [-4.0] * 3 + [0.0] * 7 + [4.0] * 5,
        atol=1e-12,
    )
    assert np.allclose(spectrum(heisenberg_ring(4)), expected, atol=1e-12)


def test_heisenberg_ring_ground_state_momentum():
    # Phase 13: The momentum of the ground state
    from eigenlab.hamiltonian import heisenberg_ring, total_spin, translation

    for n, expected_sign in [(4, 1.0), (6, -1.0), (8, 1.0)]:
        h_op = heisenberg_ring(n)
        evals, evecs = eigensystem(h_op)

        # Ground state is nondegenerate
        assert evals[1] - evals[0] > 1e-6
        gs = evecs[:, 0]

        if n == 4:
            assert np.isclose(evals[0], -8.0, atol=1e-12)

        # <S^2> = 0
        s2_op = total_spin(n)
        assert np.isclose(s2_op.expectation(gs), 0.0, atol=1e-12)

        # T eigenvalue is (-1)^{n/2}
        t_mat = translation(n)
        assert np.allclose(t_mat @ gs, expected_sign * gs, atol=1e-12)
        assert np.isclose(np.vdot(gs, t_mat @ gs).real, expected_sign, atol=1e-12)








