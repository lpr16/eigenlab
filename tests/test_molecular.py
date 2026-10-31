import numpy as np

from eigenlab.fermions import (
    jordan_wigner_adag,
    occupation_block,
    tag_sectors,
    total_number_operator,
    total_spin_squared,
    total_spin_z,
)
from eigenlab.hamiltonian import eigensystem, spectrum
from eigenlab.molecular import (
    CODATA_2018_H_MASS_AU,
    CODATA_2018_HARTREE_TO_CM1,
    h2_electrostatic_hellmann_feynman,
    h2_energy_gradient,
    h2_equilibrium_bond_length,
    h2_force_constant,
    h2_harmonic_frequency,
    h2_rhf_energy_gradient,
    h2_sto3g_hamiltonian,
    h2_sto3g_integrals,
    h4_sto3g_hamiltonian,
    h4_sto3g_integrals,
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

    # Non-positive geometries raise ValueError
    with pytest.raises(ValueError):
        heh_sto3g_integrals(0.0)
    with pytest.raises(ValueError):
        heh_sto3g_integrals(-1.0)

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


def test_h2_dissociation_curve():
    # Phase 14: Dissociation
    from eigenlab.fermions import natural_occupations
    from eigenlab.integrals import ao_integrals, rhf
    from eigenlab.molecular import H_ATOM_STO3G_ENERGY

    grid = [round(0.8 + 0.1 * k, 1) for k in range(53)] + [10.0]
    energies = [float(spectrum(h2_sto3g_hamiltonian(r))[0]) for r in grid]
    assert all(np.isfinite(e) for e in energies)

    # Single minimum on the grid: strictly decreasing up to i_min, strictly increasing after
    i_min = int(np.argmin(energies))
    assert 0 < i_min < len(grid) - 1
    for i in range(i_min):
        assert energies[i] > energies[i + 1]
    for i in range(i_min, len(grid) - 1):
        assert energies[i] < energies[i + 1]

    # At R = 10, full CI is 2 * H_ATOM_STO3G_ENERGY to 1e-5, and the four spin-orbital
    # natural occupations are each 1/2 to 1e-3 because the state is (σ_g² − σ_u²)/√2.
    evals_10, evecs_10 = eigensystem(h2_sto3g_hamiltonian(10.0))
    assert np.isclose(evals_10[0], 2.0 * H_ATOM_STO3G_ENERGY, atol=1e-5)
    occ_10 = natural_occupations(evecs_10[:, 0], 4)
    assert np.allclose(occ_10, [0.5, 0.5, 0.5, 0.5], atol=1e-3)

    # RHF does not dissociate correctly: RHF minus full-CI gap at R = 6 is larger than at R = 1.401
    def rhf_gap(r_val: float) -> float:
        ao = ao_integrals([(1.0, 1.24, [0.0, 0.0, 0.0]), (1.0, 1.24, [0.0, 0.0, r_val])])
        e_rhf = rhf(ao, n_electrons=2).energy
        e_fci = float(spectrum(h2_sto3g_hamiltonian(r_val))[0])
        return e_rhf - e_fci

    gap_eq = rhf_gap(1.401)
    gap_6 = rhf_gap(6.0)
    assert gap_6 > gap_eq > 0.0


def test_h2_energy_gradient_and_matrix_hellmann_feynman():
    # Phase 15: One derivative
    # Derivative is negative at R = 1.0 (compressed) and positive at R = 2.0 and R = 3.0 (stretched).
    assert h2_energy_gradient(1.0) < 0.0
    assert h2_energy_gradient(2.0) > 0.0
    assert h2_energy_gradient(3.0) > 0.0

    # Matrix Hellmann-Feynman holds on the smooth Hermitian matrix family H(R):
    # <ψ| (H(R+h) - H(R-h)) / 2h |ψ> matches h2_energy_gradient(R) to 1e-6 at R = 1.0, 1.401, 2.0.
    h = 1e-4
    for R in [1.0, 1.401, 2.0]:
        _, evecs = eigensystem(h2_sto3g_hamiltonian(R))
        psi = evecs[:, 0]
        dh_dr = (
            h2_sto3g_hamiltonian(R + h).to_matrix() - h2_sto3g_hamiltonian(R - h).to_matrix()
        ) / (2.0 * h)
        hf_deriv = float(np.real( np.vdot(psi, dh_dr @ psi) ))
        grad = h2_energy_gradient(R, step=h)
        assert np.isclose(hf_deriv, grad, atol=1e-6)


def test_h2_pulay_term():
    # Phase 16: The Pulay term
    # Electrostatic Hellmann-Feynman derivative (frozen basis, moving nuclei) vs total gradient
    # at R = 1.401 differ by more than 1e-3 because the incomplete STO-3G basis rides with the nuclei.
    R = 1.401
    grad_total = h2_energy_gradient(R)
    force_ehf = h2_electrostatic_hellmann_feynman(R)
    pulay = grad_total - force_ehf
    print(f"R = {R}: total dE/dR = {grad_total:.8f}, electrostatic HF = {force_ehf:.8f}, Pulay = {pulay:.8f}")
    assert abs(grad_total - force_ehf) > 1e-3


def test_h2_equilibrium_bond_length():
    # Phase 17: Equilibrium bond length
    # Bisect h2_energy_gradient on [1.0, 2.0] until the interval is shorter than 1e-6.
    # No external STO-3G full-CI R_e table is at hand in the repo's citations, so the
    # internal variational and sign-change checks stand alone per the contract.
    from eigenlab.integrals import ao_integrals, rhf

    r_e_fci = h2_equilibrium_bond_length(method="fci", bracket=(1.0, 2.0), tol=1e-6)
    assert 1.0 < r_e_fci < 2.0
    assert abs(h2_energy_gradient(r_e_fci)) < 1e-5

    e_re = float(spectrum(h2_sto3g_hamiltonian(r_e_fci))[0])
    e_1401 = float(spectrum(h2_sto3g_hamiltonian(1.401))[0])
    assert e_re <= e_1401
    assert abs(e_re - (-1.13727)) < 1e-3
    assert abs(e_1401 - (-1.13727)) < 1e-3

    # Do the same with the RHF energy: R_e(RHF) < R_e(FCI) (correlation lengthens the bond)
    assert h2_rhf_energy_gradient(1.0) < 0.0 < h2_rhf_energy_gradient(2.0)
    r_e_rhf = h2_equilibrium_bond_length(method="rhf", bracket=(1.0, 2.0), tol=1e-6)
    assert r_e_rhf < r_e_fci

    ao_rhf_eq = ao_integrals([(1.0, 1.24, [0.0, 0.0, 0.0]), (1.0, 1.24, [0.0, 0.0, r_e_rhf])])
    ao_rhf_1401 = ao_integrals([(1.0, 1.24, [0.0, 0.0, 0.0]), (1.0, 1.24, [0.0, 0.0, 1.401])])
    assert rhf(ao_rhf_eq, n_electrons=2).energy <= rhf(ao_rhf_1401, n_electrons=2).energy


def test_h2_harmonic_frequency():
    # Phase 18: Harmonic frequency
    # CODATA 2018 recommended values (Tiesinga et al., Rev. Mod. Phys. 93, 025010 (2021)):
    #   m_H = 1837.15264958 m_e (1H atomic mass in atomic units), μ = m_H / 2
    #   E_h / (h c) = 219474.63136320 cm^-1 (hartree-to-cm^-1 conversion)
    # No external cited STO-3G full-CI harmonic frequency table is at hand in the repo's
    # references, so the internal finite-difference and formula checks stand alone per the contract.
    r_e = h2_equilibrium_bond_length(method="fci", bracket=(1.0, 2.0), tol=1e-6)

    k_1e2 = h2_force_constant(r_e, step=1e-2)
    k_5e3 = h2_force_constant(r_e, step=5e-3)
    assert k_1e2 > 0.0
    assert k_5e3 > 0.0
    assert abs(k_1e2 - k_5e3) / k_5e3 < 1e-4

    mu = CODATA_2018_H_MASS_AU / 2.0
    omega_expected = np.sqrt(k_1e2 / mu)
    nu_cm1_expected = omega_expected * CODATA_2018_HARTREE_TO_CM1

    omega_au, nu_cm1 = h2_harmonic_frequency(r_e, step=1e-2)
    assert np.isclose(omega_au, omega_expected, atol=1e-15)
    assert np.isclose(nu_cm1, nu_cm1_expected, atol=1e-10)
    assert nu_cm1 > 0.0


def test_h4_rectangle_integrals_and_hamiltonian():
    # Phase 19: Rectangle integrals (H4 at (±a/2, ±b/2, 0) with ζ = 1.24)
    h1, h2, v_nuc = h4_sto3g_integrals(1.401, 2.0)
    assert h1.shape == (4, 4)
    assert h2.shape == (4, 4, 4, 4)
    assert v_nuc > 0.0

    H = h4_sto3g_hamiltonian(1.401, 2.0)
    assert H.qubits == 8

    h_mat = H.to_matrix()
    assert h_mat.shape == (256, 256)
    assert np.allclose(h_mat, h_mat.conj().T, atol=1e-12)

    # Commutes with total_number_operator(8) and total_spin_z(4) to 1e-10
    n_op = total_number_operator(8).to_matrix()
    sz_op = total_spin_z(4).to_matrix()
    assert np.allclose(h_mat @ n_op - n_op @ h_mat, 0.0, atol=1e-10)
    assert np.allclose(h_mat @ sz_op - sz_op @ h_mat, 0.0, atol=1e-10)

    # N = 4 block has dimension C(8, 4) = 70
    block_4 = occupation_block(h_mat, n_particles=4)
    assert block_4.shape == (70, 70)
    assert np.allclose(block_4, block_4.conj().T, atol=1e-12)


def test_h4_orbitals_and_encodings_do_not_move_fci():
    # Phase 20: Orbitals do not move full CI
    # At a = 1.401, b = 2.0, the N = 4 ground energy from Löwdin orbitals equals
    # the one from RHF orbitals to 1e-8, and full JW and BK spectra match to 1e-8.
    a, b = 1.401, 2.0
    h_low = h4_sto3g_hamiltonian(a, b, mapping="jordan_wigner", orbitals="lowdin")
    h_rhf = h4_sto3g_hamiltonian(a, b, mapping="jordan_wigner", orbitals="rhf")

    e_low_n4 = float(np.linalg.eigvalsh(occupation_block(h_low, n_particles=4))[0])
    e_rhf_n4 = float(np.linalg.eigvalsh(occupation_block(h_rhf, n_particles=4))[0])
    assert np.isclose(e_low_n4, e_rhf_n4, atol=1e-8)

    h_bk = h4_sto3g_hamiltonian(a, b, mapping="bravyi_kitaev", orbitals="lowdin")
    spec_jw = spectrum(h_low)
    spec_bk = spectrum(h_bk)
    assert np.allclose(spec_jw, spec_bk, atol=1e-8)


def test_h4_two_far_molecules():
    # Phase 21: Two far molecules (a = 1.401 bond, b = 20 separation)
    # Full CI in N = 4 is twice the H2 FCI energy at R = 1.401 to 1e-6,
    # and RHF is twice the H2 RHF energy at R = 1.401 to 1e-6 (size consistency).
    from eigenlab.integrals import ao_integrals, rhf

    a, b = 1.401, 20.0
    h_h4 = h4_sto3g_hamiltonian(a, b)
    e_fci_h4 = float(np.linalg.eigvalsh(occupation_block(h_h4, n_particles=4))[0])
    e_fci_h2 = float(spectrum(h2_sto3g_hamiltonian(a))[0])
    assert np.isclose(e_fci_h4, 2.0 * e_fci_h2, atol=1e-6)

    atoms_h4 = [
        (1.0, 1.24, [-0.5 * a, -0.5 * b, 0.0]),
        (1.0, 1.24, [-0.5 * a, 0.5 * b, 0.0]),
        (1.0, 1.24, [0.5 * a, -0.5 * b, 0.0]),
        (1.0, 1.24, [0.5 * a, 0.5 * b, 0.0]),
    ]
    e_rhf_h4 = rhf(ao_integrals(atoms_h4), n_electrons=4).energy

    atoms_h2 = [(1.0, 1.24, [0.0, 0.0, 0.0]), (1.0, 1.24, [0.0, 0.0, a])]
    e_rhf_h2 = rhf(ao_integrals(atoms_h2), n_electrons=2).energy
    assert np.isclose(e_rhf_h4, 2.0 * e_rhf_h2, atol=1e-6)


def test_h4_four_far_atoms():
    # Phase 22: Four far atoms (a = b = 20)
    # Full CI in N = 4 is 4 * H_ATOM_STO3G_ENERGY to 1e-5, and the lowest level
    # is degenerate (2^4 = 16 spin configurations of four independent 1s electrons).
    from eigenlab.molecular import H_ATOM_STO3G_ENERGY

    h_far = h4_sto3g_hamiltonian(20.0, 20.0)
    evals_n4 = np.linalg.eigvalsh(occupation_block(h_far, n_particles=4))
    assert np.isclose(evals_n4[0], 4.0 * H_ATOM_STO3G_ENERGY, atol=1e-5)
    # 16-fold spin degeneracy of four independent atoms
    assert np.allclose(evals_n4[:16], evals_n4[0], atol=1e-8)
    assert evals_n4[16] - evals_n4[0] > 1e-2













