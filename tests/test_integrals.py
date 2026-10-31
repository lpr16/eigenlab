import numpy as np

from eigenlab.integrals import (
    STO3G_1S_COEFFICIENTS,
    STO3G_1S_EXPONENTS,
    ao_integrals,
    boys0,
    contracted_eri,
    contracted_kinetic,
    contracted_nuclear_attraction,
    contracted_overlap,
    gaussian_eri,
    gaussian_kinetic,
    gaussian_nuclear_attraction,
    gaussian_overlap,
    h2_symmetry_orbitals,
    mo_transform,
    sto3g_1s,
)
from eigenlab.molecular import H_ATOM_STO3G_ENERGY, h2_sto3g_integrals


def test_gaussian_overlap():
    # Phase 1: Overlap of two normalized s primitives
    # Normalization: N(a) = (2a / π)^{3/4} so ∫ N(a)^2 exp(−2a |r − A|²) d³r = (2a/π)^{3/2} (π/2a)^{3/2} = 1.
    # Product of two Gaussians (Gaussian product theorem):
    #   exp(−a|r − A|²) exp(−b|r − B|²) = exp(−μ|A − B|²) exp(−p|r − P|²)
    # with p = a + b, μ = ab / p, and P = (aA + bB) / p.
    # Integrating over r ∈ R³ gives (π / p)^{3/2}. Multiplying by N(a) N(b) = (4ab / π²)^{3/4}:
    #   S = (4ab / π²)^{3/4} (π / p)^{3/2} exp(−μ|A − B|²)
    #     = (4ab / p²)^{3/4} exp(−μ|A − B|²).

    origin = np.array([0.0, 0.0, 0.0])

    # 1. Self-overlap is 1 for every exponent a > 0 and any centre
    for a in [0.1, 0.405771, 1.0, 2.22766, 7.5]:
        for center in [origin, np.array([1.2, -0.5, 0.8])]:
            assert np.isclose(gaussian_overlap(a, center, a, center), 1.0, atol=1e-15)

    # 2. Direct comparison with the Gaussian product formula
    a, b = 0.4, 1.3
    A = np.array([0.1, -0.2, 0.5])
    B = np.array([-0.3, 0.7, 1.4])
    p = a + b
    mu = a * b / p
    r2 = float(np.sum((A - B) ** 2))
    expected = ((4.0 * a * b) / (p * p)) ** 0.75 * np.exp(-mu * r2)
    assert np.isclose(gaussian_overlap(a, A, b, B), expected, atol=1e-15)

    # 3. Symmetry in its two arguments: S(a, A; b, B) == S(b, B; a, A)
    assert np.isclose(gaussian_overlap(a, A, b, B), gaussian_overlap(b, B, a, A), atol=1e-15)

    # 4. Equals 1 only when a = b and A = B:
    #    4ab / (a + b)² = 1 − (a − b)² / (a + b)² < 1 whenever a != b,
    #    and exp(−μ|A − B|²) < 1 whenever A != B.
    assert gaussian_overlap(0.5, origin, 1.2, origin) < 1.0
    assert gaussian_overlap(0.5, origin, 0.5, np.array([0.0, 0.0, 0.2])) < 1.0
    assert gaussian_overlap(0.5, origin, 1.2, np.array([0.0, 0.0, 0.2])) < 1.0

    # 5. Strictly decreases as |A − B| grows
    distances = [0.0, 0.5, 1.0, 1.5, 2.5, 4.0]
    overlaps = [
        gaussian_overlap(a, origin, b, np.array([0.0, 0.0, d]))
        for d in distances
    ]
    for s_near, s_far in zip(overlaps, overlaps[1:]):
        assert s_near > s_far > 0.0


def test_sto3g_1s_contraction():
    # Phase 2: STO-3G 1s contraction
    # Reference: Hehre, Stewart, and Pople, J. Chem. Phys. 51, 2657 (1969), Table I.
    # For ζ = 1: exponents (0.109818, 0.405771, 2.22766), coefficients (0.444635, 0.535328, 0.154329).
    # Exponents scale as ζ²; contraction coefficients do not.
    chi_unit = sto3g_1s(1.0, [0.0, 0.0, 0.0])
    assert chi_unit.exponents == (0.109818, 0.405771, 2.22766)
    assert chi_unit.coefficients == (0.444635, 0.535328, 0.154329)
    assert STO3G_1S_EXPONENTS == (0.109818, 0.405771, 2.22766)
    assert STO3G_1S_COEFFICIENTS == (0.444635, 0.535328, 0.154329)

    for zeta in [1.0, 1.24, 2.0925]:
        chi = sto3g_1s(zeta, [0.2, -0.1, 0.5])
        assert np.allclose(chi.exponents, [a * zeta**2 for a in STO3G_1S_EXPONENTS], atol=1e-15)
        assert chi.coefficients == STO3G_1S_COEFFICIENTS
        # Contracted self-overlap is 1 to 1e-5 (six-digit printed coefficients)
        assert np.isclose(contracted_overlap(chi, chi), 1.0, atol=1e-5)

    # At R = 1.4632 bohr with ζ_He = 2.0925 and ζ_H = 1.24, the contracted overlap
    # matches S_12 = 0.450770 transcribed in heh_sto3g_integrals docstring (Szabo & Ostlund Table 3.5).
    R = 1.4632
    chi_he = sto3g_1s(2.0925, [0.0, 0.0, 0.0])
    chi_h = sto3g_1s(1.24, [0.0, 0.0, R])
    s_12 = contracted_overlap(chi_he, chi_h)
    assert np.isclose(s_12, 0.450770, atol=1e-5)


def test_kinetic_energy_integral():
    # Phase 3: Kinetic energy integral for s primitives and STO-3G contractions
    # Algebra:
    #   −½ ∇² exp(−b|r − B|²) = b (3 − 2b|r − B|²) exp(−b|r − B|²).
    # In the Gaussian product centre P = (aA + bB)/p with p = a + b and μ = ab/p,
    #   r − B = (r − P) + (a/p)(A − B),
    #   ⟨|r − P|²⟩_P = 3 / (2p), ⟨r − P⟩_P = 0, so ⟨|r − B|²⟩_P = 3/(2p) + (a²/p²)|A − B|².
    # Hence:
    #   T = b [ 3 − 2b (3/(2p) + (a²/p²)|A − B|²) ] S
    #     = [ 3ab/p − 2(ab/p)² |A − B|² ] S
    #     = μ (3 − 2μ|A − B|²) S.
    # For one normalized primitive at one centre (a = b, A = B):
    #   S = 1, μ = a² / (2a) = a / 2, |A − B|² = 0  ==>  T = 3 μ = 3a / 2.

    for a in [0.15, 0.5, 1.0, 2.22766, 4.5]:
        for center in [[0.0, 0.0, 0.0], [1.1, -0.4, 2.3]]:
            assert np.isclose(gaussian_kinetic(a, center, a, center), 1.5 * a, atol=1e-14)

    # Check two distinct primitives against the formula and symmetry
    a, b = 0.405771, 1.35
    A = np.array([0.2, -0.3, 0.1])
    B = np.array([-0.5, 0.4, 1.2])
    p = a + b
    mu = a * b / p
    r2 = float(np.sum((A - B) ** 2))
    s = gaussian_overlap(a, A, b, B)
    expected_t = mu * (3.0 - 2.0 * mu * r2) * s
    assert np.isclose(gaussian_kinetic(a, A, b, B), expected_t, atol=1e-15)
    assert np.isclose(gaussian_kinetic(a, A, b, B), gaussian_kinetic(b, B, a, A), atol=1e-15)

    # Contracted kinetic integral is symmetric and positive on the diagonal
    chi_he = sto3g_1s(2.0925, [0.0, 0.0, 0.0])
    chi_h = sto3g_1s(1.24, [0.0, 0.0, 1.4632])
    assert contracted_kinetic(chi_he, chi_he) > 0.0
    assert contracted_kinetic(chi_h, chi_h) > 0.0
    assert np.isclose(contracted_kinetic(chi_he, chi_h), contracted_kinetic(chi_h, chi_he), atol=1e-15)


def test_boys0_and_nuclear_attraction():
    # Phase 4: Boys function and nuclear attraction
    # 1. boys0(0) = 1, and for small t the Taylor series 1 − t/3 + t²/10 matches to 1e-10 at t = 1e-3.
    assert boys0(0.0) == 1.0
    t = 1e-3
    series = 1.0 - t / 3.0 + (t * t) / 10.0
    assert np.isclose(boys0(t), series, atol=1e-10)

    # 2. One normalized primitive with the nucleus at its centre (a = b, A = B = C):
    #    p = 2a, μ = a/2, K = exp(0) = 1, P = A = C so F0(0) = 1, and N(a)² = (2a/π)^{3/2}.
    #    V = −Z (2π / 2a) (1) (1) (2a / π)^{3/2} = −Z (π / a) (2a / π)^{3/2} = −2Z √(2a / π).
    for a in [0.15, 0.5, 1.0, 2.22766]:
        for z in [1.0, 2.0]:
            center = [0.3, -0.2, 0.7]
            expected_v = -2.0 * z * np.sqrt(2.0 * a / np.pi)
            assert np.isclose(
                gaussian_nuclear_attraction(a, center, a, center, z, center),
                expected_v,
                atol=1e-14,
            )

    # 3. STO-3G hydrogen atom (ζ = 1.24, Z = 1) has T + V equal to H_ATOM_STO3G_ENERGY (-0.466582) to 1e-5.
    chi_h_atom = sto3g_1s(1.24, [0.0, 0.0, 0.0])
    e_h_atom = contracted_kinetic(chi_h_atom, chi_h_atom) + contracted_nuclear_attraction(
        chi_h_atom, chi_h_atom, 1.0, [0.0, 0.0, 0.0]
    )
    assert np.isclose(e_h_atom, H_ATOM_STO3G_ENERGY, atol=1e-5)

    # 4. HeH+ one-electron integrals h_11, h_12, h_22 at R = 1.4632 match heh_sto3g_integrals docstring
    #    (Szabo & Ostlund Section 3.5.3, Table 3.5: h_11 = -2.652744, h_12 = -1.347205, h_22 = -1.731828).
    r_he = [0.0, 0.0, 0.0]
    r_h = [0.0, 0.0, 1.4632]
    chi_he = sto3g_1s(2.0925, r_he)
    chi_h = sto3g_1s(1.24, r_h)

    def core_elem(ca, cb):
        return (
            contracted_kinetic(ca, cb)
            + contracted_nuclear_attraction(ca, cb, 2.0, r_he)
            + contracted_nuclear_attraction(ca, cb, 1.0, r_h)
        )

    h_11 = core_elem(chi_he, chi_he)
    h_12 = core_elem(chi_he, chi_h)
    h_21 = core_elem(chi_h, chi_he)
    h_22 = core_elem(chi_h, chi_h)

    assert np.isclose(h_12, h_21, atol=1e-15)
    assert np.isclose(h_11, -2.652744, atol=1e-5)
    assert np.isclose(h_12, -1.347205, atol=1e-5)
    assert np.isclose(h_22, -1.731828, atol=1e-5)


def test_electron_repulsion_integral():
    # Phase 5: Electron repulsion integral (ab|cd) in chemist notation
    # Algebra for four copies of one normalized primitive at one centre (a = b = c = d, A = B = C = D):
    #   p = 2a, q = 2a, p + q = 4a, K_ab = K_cd = 1, P = Q so F0(0) = 1,
    #   and N(a)⁴ = (2a / π)³ = 8a³ / π³.
    #   (aa|aa) = [ 2π^{5/2} / ((2a)(2a)√(4a)) ] · (8a³ / π³)
    #           = [ 2π^{5/2} / (8a^{5/2}) ] · (8a³ / π³)
    #           = 2 √(a / π).
    for a in [0.15, 0.5, 1.0, 2.22766, 3.8]:
        center = [0.2, -0.4, 0.9]
        expected = 2.0 * np.sqrt(a / np.pi)
        assert np.isclose(
            gaussian_eri(a, center, a, center, a, center, a, center),
            expected,
            atol=1e-14,
        )

    # Real eightfold symmetry:
    # (ab|cd) = (ba|cd) = (ab|dc) = (ba|dc) = (cd|ab) = (dc|ab) = (cd|ba) = (dc|ba)
    p1 = (0.3, [0.0, 0.1, 0.2])
    p2 = (0.7, [0.4, -0.2, 0.5])
    p3 = (1.1, [-0.3, 0.6, 0.1])
    p4 = (1.6, [0.2, 0.3, -0.4])
    ref = gaussian_eri(*p1, *p2, *p3, *p4)
    for perm in [
        (p1, p2, p3, p4),
        (p2, p1, p3, p4),
        (p1, p2, p4, p3),
        (p2, p1, p4, p3),
        (p3, p4, p1, p2),
        (p4, p3, p1, p2),
        (p3, p4, p2, p1),
        (p4, p3, p2, p1),
    ]:
        val = gaussian_eri(*perm[0], *perm[1], *perm[2], *perm[3])
        assert np.isclose(val, ref, atol=1e-15)

    # Six HeH+ AO two-electron integrals at R = 1.4632 match heh_sto3g_integrals docstring
    # (Szabo & Ostlund Section 3.5.3, Table 3.5):
    #   (11|11) = 1.307152, (21|11) = 0.437279, (21|21) = 0.177267,
    #   (22|11) = 0.605703, (22|21) = 0.311795, (22|22) = 0.774608.
    chi_1 = sto3g_1s(2.0925, [0.0, 0.0, 0.0])
    chi_2 = sto3g_1s(1.24, [0.0, 0.0, 1.4632])

    assert np.isclose(contracted_eri(chi_1, chi_1, chi_1, chi_1), 1.307152, atol=1e-5)
    assert np.isclose(contracted_eri(chi_2, chi_1, chi_1, chi_1), 0.437279, atol=1e-5)
    assert np.isclose(contracted_eri(chi_2, chi_1, chi_2, chi_1), 0.177267, atol=1e-5)
    assert np.isclose(contracted_eri(chi_2, chi_2, chi_1, chi_1), 0.605703, atol=1e-5)
    assert np.isclose(contracted_eri(chi_2, chi_2, chi_2, chi_1), 0.311795, atol=1e-5)
    assert np.isclose(contracted_eri(chi_2, chi_2, chi_2, chi_2), 0.774608, atol=1e-5)


def test_ao_integrals_bundle_heh():
    # Phase 6: The AO bundle
    # For HeH+ at R = 1.4632 (Z_He = 2, ζ_He = 2.0925; Z_H = 1, ζ_H = 1.24),
    # every transcribed number from phases 2–5 comes out of a single ao_integrals call.
    R = 1.4632
    atoms = [
        (2.0, 2.0925, [0.0, 0.0, 0.0]),
        (1.0, 1.24, [0.0, 0.0, R]),
    ]
    S, h, eri, v_nuc = ao_integrals(atoms)

    # Symmetry checks
    assert S.shape == (2, 2)
    assert h.shape == (2, 2)
    assert eri.shape == (2, 2, 2, 2)
    assert np.allclose(S, S.T, atol=1e-15)
    assert np.allclose(h, h.T, atol=1e-15)
    for p in range(2):
        for q in range(2):
            for r in range(2):
                for s in range(2):
                    val = eri[p, q, r, s]
                    assert np.isclose(eri[q, p, r, s], val, atol=1e-15)
                    assert np.isclose(eri[p, q, s, r], val, atol=1e-15)
                    assert np.isclose(eri[r, s, p, q], val, atol=1e-15)

    # Nuclear repulsion V_nuc = 2 / R
    assert np.isclose(v_nuc, 2.0 / R, atol=1e-15)

    # Overlap S_11 = S_22 = 1, S_12 = 0.450770
    assert np.isclose(S[0, 0], 1.0, atol=1e-5)
    assert np.isclose(S[1, 1], 1.0, atol=1e-5)
    assert np.isclose(S[0, 1], 0.450770, atol=1e-5)

    # Core Hamiltonian h_11 = -2.652744, h_12 = -1.347205, h_22 = -1.731828
    assert np.isclose(h[0, 0], -2.652744, atol=1e-5)
    assert np.isclose(h[0, 1], -1.347205, atol=1e-5)
    assert np.isclose(h[1, 1], -1.731828, atol=1e-5)

    # Six unique two-electron integrals in chemist notation
    assert np.isclose(eri[0, 0, 0, 0], 1.307152, atol=1e-5)
    assert np.isclose(eri[1, 0, 0, 0], 0.437279, atol=1e-5)
    assert np.isclose(eri[1, 0, 1, 0], 0.177267, atol=1e-5)
    assert np.isclose(eri[1, 1, 0, 0], 0.605703, atol=1e-5)
    assert np.isclose(eri[1, 1, 1, 0], 0.311795, atol=1e-5)
    assert np.isclose(eri[1, 1, 1, 1], 0.774608, atol=1e-5)


def test_minimal_basis_orbitals_and_mo_transform():
    # Phase 7: Minimal-basis orbitals
    # For two hydrogen 1s functions φ_1, φ_2 with overlap S:
    #   σ_g = (φ_1 + φ_2) / √(2(1 + S)),   σ_u = (φ_1 − φ_2) / √(2(1 − S))
    # With C = [σ_g, σ_u], Cᵀ S C = I to 1e-12.
    for R in [1.0, 1.401, 2.0, 3.0]:
        atoms = [(1.0, 1.24, [0.0, 0.0, 0.0]), (1.0, 1.24, [0.0, 0.0, R])]
        S, h, eri, _ = ao_integrals(atoms)

        # Direct check with unit-diagonal overlap matrix [[1, s], [s, 1]] and formula 1/√(2(1±s))
        s = S[0, 1]
        S_unit = np.array([[1.0, s], [s, 1.0]])
        C_unit = np.array(
            [
                [1.0 / np.sqrt(2.0 * (1.0 + s)), 1.0 / np.sqrt(2.0 * (1.0 - s))],
                [1.0 / np.sqrt(2.0 * (1.0 + s)), -1.0 / np.sqrt(2.0 * (1.0 - s))],
            ]
        )
        assert np.allclose(C_unit.T @ S_unit @ C_unit, np.eye(2), atol=1e-12)

        # And with the AO overlap matrix S directly via h2_symmetry_orbitals(S)
        C = h2_symmetry_orbitals(S)
        assert np.allclose(C.T @ S @ C, np.eye(2), atol=1e-12)
        assert np.allclose(C, C_unit, atol=1e-5)

        h_mo, eri_mo = mo_transform(C, h, eri)
        assert h_mo.shape == (2, 2)
        assert eri_mo.shape == (2, 2, 2, 2)
        assert np.allclose(h_mo, h_mo.T, atol=1e-14)
        # By inversion symmetry, <σ_g | h | σ_u> = 0
        assert np.isclose(h_mo[0, 1], 0.0, atol=1e-14)


def test_h2_table_reproduced():
    # Phase 8: The table, reproduced
    # At R = 1.0, 1.401, 2.0, 3.0, the MO integrals from phase 7 match the
    # h2_sto3g_integrals table to 1e-5, entry by entry.
    for R in [1.0, 1.401, 2.0, 3.0]:
        atoms = [(1.0, 1.24, [0.0, 0.0, 0.0]), (1.0, 1.24, [0.0, 0.0, R])]
        S, h, eri, v_nuc = ao_integrals(atoms)
        C = h2_symmetry_orbitals(S)
        h_mo, eri_mo = mo_transform(C, h, eri)

        h1_tab, h2_tab, v_nuc_tab = h2_sto3g_integrals(R)
        assert np.isclose(v_nuc, v_nuc_tab, atol=1e-12)
        assert np.allclose(h_mo, h1_tab, atol=1e-5)
        assert np.allclose(eri_mo, h2_tab, atol=1e-5)







