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


def test_degenerate_perturbation():
    # Phase 45: Degenerate perturbation
    # H = XX + YY + ZZ + ε ZZ is the XXZ dimer at Δ = 1 + ε.
    # The triplet |00⟩, |11⟩, (|01⟩ + |10⟩)/√2 sits at +1. Inside it, ZZ is diagonal with
    # entries +1, +1, −1, so first-order degenerate theory gives 1 + ε, 1 + ε, 1 − ε.
    # ZZ commutes with XX + YY + ZZ, so these are exact. The singlet (|01⟩ − |10⟩)/√2 at −3
    # has ⟨ZZ⟩ = −1 and moves alone to −3 − ε.
    from eigenlab.hamiltonian import heisenberg_dimer, xxz_dimer
    from eigenlab.pauli import pauli
    from eigenlab.states import basis, bell

    h0 = heisenberg_dimer()
    zz = pauli("ZZ")
    triplet = np.column_stack(
        [basis(0, 2), basis(3, 2), (basis(1, 2) + basis(2, 2)) / np.sqrt(2.0)]
    )
    singlet = bell("psi-")

    assert np.allclose(triplet.conj().T @ h0 @ triplet, np.eye(3), atol=1e-12)
    assert np.isclose(np.vdot(singlet, h0 @ singlet).real, -3.0, atol=1e-12)

    # The perturbation inside the degenerate subspace.
    w = triplet.conj().T @ zz @ triplet
    assert np.allclose(w, np.diag([1.0, 1.0, -1.0]), atol=1e-12)

    for eps in [0.1, -0.05, 0.3]:
        first_order = np.sort(1.0 + eps * np.linalg.eigvalsh(w))
        assert np.allclose(first_order, np.sort([1.0 + eps, 1.0 + eps, 1.0 - eps]), atol=1e-12)
        singlet_energy = -3.0 + eps * np.vdot(singlet, zz @ singlet).real
        assert np.isclose(singlet_energy, -3.0 - eps, atol=1e-12)
        assert np.allclose(
            spectrum(xxz_dimer(1.0 + eps)),
            np.sort(np.concatenate([[singlet_energy], first_order])),
            atol=1e-12,
        )
