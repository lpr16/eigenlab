import numpy as np

from eigenlab.hamiltonian import (
    Hamiltonian,
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
