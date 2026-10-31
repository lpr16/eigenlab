import numpy as np

from eigenlab.integrals import (
    STO3G_1S_COEFFICIENTS,
    STO3G_1S_EXPONENTS,
    contracted_overlap,
    gaussian_overlap,
    sto3g_1s,
)


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

