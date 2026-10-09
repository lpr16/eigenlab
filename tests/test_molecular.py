import numpy as np

from eigenlab.fermions import (
    jordan_wigner_adag,
    tag_sectors,
    total_spin_squared,
)
from eigenlab.hamiltonian import eigensystem, spectrum
from eigenlab.molecular import (
    h2_sto3g_hamiltonian,
    h2_sto3g_integrals,
    heh_sto3g_hamiltonian,
    heh_sto3g_integrals,
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


def test_h2_sto3g_potential_curve_geometries():
    r"""Phase 20: Shorter and stretched bonds, and dissociation toward isolated atoms.

    Tests three geometries along the H2 STO-3G potential curve from the same
    citation family (Szabo & Ostlund Appendix D / Whitfield et al. / O'Malley et al.):
    - Shorter bond: R = 1.0 a.u., E_FCI = -1.07897 Hartree
    - Equilibrium bond: R = 1.401 a.u., E_FCI = -1.13727 Hartree
    - Stretched bond: R = 3.0 a.u., E_FCI = -0.98516 Hartree

    Dissociation check:
    As the bond stretches (R = 1.401 -> 2.0 -> 3.0), the energy increases monotonically
    toward the sum of two isolated STO-3G hydrogen atoms:
    2 * E(H atom, STO-3G) = 2 * (-0.466582) = -0.933164 Hartree.
    """
    from eigenlab.molecular import H_ATOM_STO3G_ENERGY

    # Shorter bond: R = 1.0 a.u.
    H_short = h2_sto3g_hamiltonian(1.0)
    E_short = spectrum(H_short)[0]
    assert np.isclose(E_short, -1.07897, atol=1e-4)

    # Equilibrium bond: R = 1.401 a.u.
    H_eq = h2_sto3g_hamiltonian(1.401)
    E_eq = spectrum(H_eq)[0]
    assert np.isclose(E_eq, -1.13727, atol=1e-4)

    # Intermediate stretched bond: R = 2.0 a.u.
    H_mid = h2_sto3g_hamiltonian(2.0)
    E_mid = spectrum(H_mid)[0]
    assert np.isclose(E_mid, -1.08850, atol=1e-4)

    # Stretched bond: R = 3.0 a.u.
    H_stretch = h2_sto3g_hamiltonian(3.0)
    E_stretch = spectrum(H_stretch)[0]
    assert np.isclose(E_stretch, -0.98516, atol=1e-4)

    # Equilibrium geometry is the energy minimum
    assert E_eq < E_short
    assert E_eq < E_mid < E_stretch

    # Two isolated STO-3G hydrogen atoms asymptote
    E_two_atoms = 2.0 * H_ATOM_STO3G_ENERGY
    assert np.isclose(E_two_atoms, -0.933164, atol=1e-5)

    # Stretched geometry energy is bounded above by the two-atom limit and
    # moves progressively closer to it than equilibrium or intermediate bonds
    assert E_stretch < E_two_atoms
    assert abs(E_stretch - E_two_atoms) < abs(E_mid - E_two_atoms)
    assert abs(E_mid - E_two_atoms) < abs(E_eq - E_two_atoms)

    # Sector tagging remains N = 2, S_z = 0 across geometries
    for H_geom in [H_short, H_eq, H_mid, H_stretch]:
        evals, evecs = eigensystem(H_geom)
        tags = tag_sectors(evecs, n_spatial=2)
        assert tags[0][0] == 2
        assert np.isclose(tags[0][1], 0.0)


def test_h2_sto3g_bravyi_kitaev():
    # Phase 2: H2 STO-3G at R = 1.401 a.u. with Bravyi-Kitaev mapping
    R_au = 1.401
    h1, h2, v_nuc = h2_sto3g_integrals(R_au)
    h1_spin, h2_spin = spatial_to_spin_orbital(h1, h2)

    H_jw = integral_hamiltonian(h1_spin, h2_spin, nuclear_repulsion=v_nuc, mapping="jordan_wigner")
    H_bk = integral_hamiltonian(h1_spin, h2_spin, nuclear_repulsion=v_nuc, mapping="bravyi_kitaev")

    e_jw = spectrum(H_jw)[0]
    e_bk = spectrum(H_bk)[0]

    assert np.isclose(e_jw, e_bk, atol=1e-8)
    assert np.isclose(e_bk, -1.13727, atol=1e-4)

    # Check via h2_sto3g_hamiltonian mapping argument
    H_bk_direct = h2_sto3g_hamiltonian(R_au, mapping="bravyi_kitaev")
    assert np.isclose(spectrum(H_bk_direct)[0], e_jw, atol=1e-8)


def test_heh_sto3g():
    r"""Phase 7: STO-3G HeH+ ground state, sector tagging, and BK mapping.

    Literature citations:
    - Szabo & Ostlund, Modern Quantum Chemistry: Introduction to Advanced
      Electronic Structure Theory (1996), Section 3.5.3 (pp. 170-179) and Table 3.5.
    - Comparison: Wang et al., "Quantum Simulation of Helium Hydride in a
      Solid-State Spin Register", ACS Nano 9, 7769 (2015) / arXiv:1405.2696.

    Parameters:
    - Geometry: R = 1.4632 a.u. (bohr), internuclear distance approx. 0.7743 Å.
      (Wang et al. investigated R = 91.3 pm approx. 1.7253 bohr; we follow the
      classic Szabo & Ostlund Section 3.5.3 geometry at R = 1.4632 bohr).
    - Basis set: STO-3G minimal basis (He 1s and H 1s).
    - Nuclear repulsion: V_nuc = Z_He * Z_H / R = 2.0 / 1.4632 = 1.36686714... Hartree.

    Reported Energies:
    - Szabo & Ostlund Table 3.5 RHF energy:
        E_RHF_elec = -4.227529 Hartree
        E_RHF_total = -2.860662 Hartree (includes nuclear repulsion V_nuc).
    - Full Configuration Interaction (FCI) 2-electron ground state energy:
        E_FCI_elec = -4.247579 Hartree
        E_FCI_total = -2.880712 Hartree (includes nuclear repulsion V_nuc).

    Fock Space Spectrum Note:
    In the unconstrained second-quantized 4-qubit Fock space, the neutral radical
    HeH sector (N = 3) sits lower at -2.922682 Hartree because the HeH+ cation
    has positive electron affinity (virtual orbital energy eps_2 = -0.0617 < 0
    in STO-3G). The physical HeH+ cation ground state is the lowest eigenstate
    in the N = 2 sector, which is a spin singlet with N = 2, S_z = 0, and <S^2> = 0.
    """
    import pytest

    # 1. Integrals and nuclear repulsion
    R_au = 1.4632
    h1, h2, v_nuc = heh_sto3g_integrals(R_au)
    assert np.isclose(v_nuc, 2.0 / R_au)
    assert h1.shape == (2, 2)
    assert h2.shape == (2, 2, 2, 2)

    # Unsupported geometries raise ValueError (do not add a second bond length)
    with pytest.raises(ValueError):
        heh_sto3g_integrals(1.0)

    # 2. Build Hamiltonians with Jordan-Wigner and Bravyi-Kitaev mappings
    H_jw = heh_sto3g_hamiltonian(R_au, mapping="jordan_wigner")
    H_bk = heh_sto3g_hamiltonian(R_au, mapping="bravyi_kitaev")

    assert H_jw.qubits == 4
    assert H_bk.qubits == 4

    # 3. Both mappings have identical spectra to 1e-8
    spec_jw = spectrum(H_jw)
    spec_bk = spectrum(H_bk)
    assert np.isclose(spec_jw[0], spec_bk[0], atol=1e-8)
    assert np.allclose(spec_jw, spec_bk, atol=1e-8)

    # 4. Diagonalize and identify the 2-electron HeH+ cation ground state
    evals, evecs = eigensystem(H_jw)
    tags = tag_sectors(evecs, n_spatial=2)

    n2_indices = [i for i, (n, sz) in enumerate(tags) if n == 2]
    i_n2 = n2_indices[0]
    e_fci_n2 = evals[i_n2]

    # Energy matches FCI reference (-2.88071 Hartree) to reported precision
    assert np.isclose(e_fci_n2, -2.880712, atol=1e-5)

    # Sector: N = 2, S_z = 0
    assert tags[i_n2] == (2, 0.0)

    # Total spin singlet: <S^2> = 0
    psi_n2 = evecs[:, i_n2]
    s2_op = total_spin_squared(n_spatial=2)
    assert np.isclose(s2_op.expectation(psi_n2), 0.0, atol=1e-6)




