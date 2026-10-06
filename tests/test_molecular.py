import numpy as np

from eigenlab.fermions import jordan_wigner_adag, tag_sectors
from eigenlab.hamiltonian import eigensystem, spectrum
from eigenlab.molecular import (
    h2_sto3g_hamiltonian,
    h2_sto3g_integrals,
    integral_hamiltonian,
    spatial_to_spin_orbital,
)
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


def test_h2_sto3g_equilibrium_geometry():
    r"""Phase 19: STO-3G H2 ground state at equilibrium geometry.

    Literature citations:
    - Whitfield, Biamonte, Aspuru-Guzik, "Simulation of electronic structure
      Hamiltonians using quantum computers", Mol. Phys. 109, 735-750 (2011),
      arXiv:1001.3855, Table 7.
    - Seeley, Richard, Love, "The Bravyi-Kitaev transformation for quantum
      computation of electronic structure", J. Chem. Phys. 137, 224109 (2012),
      Table III.
    - O'Malley et al., "Scalable Quantum Simulation of Molecular Energies",
      Phys. Rev. X 6, 031007 (2016), Table I.
    - Szabo & Ostlund, Modern Quantum Chemistry: Introduction to Advanced
      Electronic Structure Theory (1996), Section 3.5 & Appendix D.

    Parameters:
    - Geometry: R = 1.401000 a.u. (internuclear separation approx. 0.7414 Å).
    - Basis set: STO-3G minimal basis (two 1s orbitals forming sigma_g and sigma_u).
    - Nuclear repulsion: V_nuc = 1/R = 1/1.401 = 0.71377587... Hartree is included.

    Cited FCI Ground State Energies:
    - Whitfield et al. (2011), Table 7 prints: E_FCI = -1.1373 Hartree.
    - O'Malley et al. (2016), Table I prints: E_FCI = -1.137 Hartree.
    - Szabo & Ostlund (1996), Section 3.5 prints: E_FCI = -1.13728 Hartree (at R = 1.4 a.u.).
    These cited references agree on the ground-state energy to their reported precision.
    """
    R_au = 1.401
    H = h2_sto3g_hamiltonian(R_au)

    # 4 spin orbitals -> 4 qubits
    assert H.qubits == 4

    # Diagonalize using the existing eigensolver
    evals, evecs = eigensystem(H)
    e_ground = evals[0]

    # Verify against cited full configuration interaction (FCI) energies:
    # 1. Whitfield et al. (2011) reports -1.1373 to 4 decimal places
    assert round(float(e_ground), 4) == -1.1373
    assert np.isclose(e_ground, -1.1373, atol=1e-4)

    # 2. O'Malley et al. (2016) reports -1.137 to 3 decimal places
    assert round(float(e_ground), 3) == -1.137
    assert np.isclose(e_ground, -1.137, atol=1e-3)

    # 3. High-precision comparison with transcribed numerical integrals
    # Computed value is -1.1372698...
    assert np.isclose(e_ground, -1.13727, atol=1e-5)

    # Verify sector tagging: neutral singlet ground state (N = 2, S_z = 0)
    tags = tag_sectors(evecs, n_spatial=2)
    n_elec, s_z = tags[0]
    assert n_elec == 2
    assert np.isclose(s_z, 0.0)

