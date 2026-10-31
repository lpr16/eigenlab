import numpy as np

from eigenlab.integrals import gaussian_overlap


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
