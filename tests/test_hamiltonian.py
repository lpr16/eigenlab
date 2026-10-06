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
