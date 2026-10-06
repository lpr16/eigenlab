import numpy as np

from eigenlab.fermions import jordan_wigner_adag
from eigenlab.molecular import integral_hamiltonian, spatial_to_spin_orbital
from eigenlab.pauli import expectation
from eigenlab.states import basis


def test_integral_hamiltonian_two_orbital_toy():
    # Phase 18:
    # Build a Pauli term list from 1- and 2-electron arrays in chemist notation (pq|rs).
    # Test on a two-orbital diagonal toy:
    # Computational occupation state |11⟩ on orbitals i=0 and j=1:
    # Expectation value equals h_ii + h_jj + (ii|jj) − (ij|ji)
    # computed from the integrals directly and from the term list.
    n = 2
    h_00 = 1.75
    h_11 = -0.65
    coulomb_01 = 0.85  # (00|11)
    exchange_01 = 0.35  # (01|10)

    h1 = np.zeros((n, n), dtype=float)
    h1[0, 0] = h_00
    h1[1, 1] = h_11

    h2 = np.zeros((n, n, n, n), dtype=float)
    # Chemist notation: (pq|rs)
    # Coulomb terms: (00|11) and (11|00)
    h2[0, 0, 1, 1] = coulomb_01
    h2[1, 1, 0, 0] = coulomb_01
    # Exchange terms: (01|10) and (10|01)
    h2[0, 1, 1, 0] = exchange_01
    h2[1, 0, 0, 1] = exchange_01

    # Build Pauli term list Hamiltonian
    H = integral_hamiltonian(h1, h2)

    # Computational occupation state where both orbitals 0 and 1 are occupied:
    # |ψ⟩ = a†_0 a†_1 |0⟩
    vac = basis(0, n)
    psi_11 = (jordan_wigner_adag(0, n) @ jordan_wigner_adag(1, n)).to_matrix() @ vac

    # 1. Expectation value from term list
    exp_term_list = H.expectation(psi_11)

    # 2. Expectation value computed from integrals directly:
    # ⟨H⟩ = h_ii + h_jj + (ii|jj) - (ij|ji)
    expected_direct = h_00 + h_11 + coulomb_01 - exchange_01

    assert np.isclose(exp_term_list, expected_direct)
    assert np.isclose(expected_direct, 1.75 - 0.65 + 0.85 - 0.35)
    assert np.isclose(exp_term_list, 1.60)

    # Test with off-diagonal one-body elements and general indices (i=1, j=2 in n=3 system)
    n3 = 3
    h1_3 = np.diag([0.5, 2.0, 3.5])
    h2_3 = np.zeros((3, 3, 3, 3))
    h2_3[1, 1, 2, 2] = 1.2
    h2_3[2, 2, 1, 1] = 1.2
    h2_3[1, 2, 2, 1] = 0.4
    h2_3[2, 1, 1, 2] = 0.4

    H3 = integral_hamiltonian(h1_3, h2_3)
    # State with orbitals 1 and 2 occupied
    psi_12 = (jordan_wigner_adag(1, 3) @ jordan_wigner_adag(2, 3)).to_matrix() @ basis(0, 3)
    exp3 = H3.expectation(psi_12)
    expected3 = h1_3[1, 1] + h1_3[2, 2] + h2_3[1, 1, 2, 2] - h2_3[1, 2, 2, 1]
    assert np.isclose(exp3, expected3)
    assert np.isclose(exp3, 2.0 + 3.5 + 1.2 - 0.4)
