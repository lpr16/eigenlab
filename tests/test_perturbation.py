import numpy as np

from eigenlab.hamiltonian import eigensystem, spectrum
from eigenlab.pauli import X, Z
from eigenlab.states import ONE, ZERO


def test_nondegenerate_perturbation():
    # Phase 44: Nondegenerate perturbation
    # H = Z/2 + λ X = [[1/2, λ], [λ, −1/2]], eigenvalues ±√(1/4 + λ²).
    # At λ = 0 the ground state is |1⟩ with E₀ = −1/2; the only other state is |0⟩ at +1/2.
    # Second order: E₂ = |⟨0|λX|1⟩|² / (E₀ − E_|0⟩) = λ² / (−1/2 − 1/2) = −λ².
    # −√(1/4 + λ²) = −1/2 − λ² + λ⁴ + O(λ⁶), so the polynomial is off by about λ⁴ = 10^{−12}.
    h0 = 0.5 * Z
    energies, vectors = eigensystem(h0)
    assert np.isclose(energies[0], -0.5, atol=1e-12)
    assert np.isclose(abs(np.vdot(ONE, vectors[:, 0])), 1.0, atol=1e-12)

    lam = 1e-3
    numerator = abs(np.vdot(ZERO, lam * X @ ONE)) ** 2
    gap = -0.5 - 0.5
    assert np.isclose(numerator, lam**2, atol=1e-18)
    assert gap == -1.0
    second_order = -0.5 + numerator / gap

    exact = spectrum(h0 + lam * X)
    assert np.allclose(exact, [-np.sqrt(0.25 + lam**2), np.sqrt(0.25 + lam**2)], atol=1e-14)
    assert abs(exact[0] - second_order) < 1e-11
