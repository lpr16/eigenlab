import numpy as np

from eigenlab.hamiltonian import (
    Hamiltonian,
    eigensystem,
    heisenberg_dimer,
    heisenberg_dimer_terms,
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
