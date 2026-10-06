import numpy as np

import pytest

from eigenlab.states import (
    ONE,
    PLUS,
    PLUS_I,
    ZERO,
    basis,
    bell,
    bloch,
    density,
    partial_trace,
    probabilities,
    purity,
)


def test_phi_plus_is_correlated():
    probs = probabilities(bell("phi+"))
    assert np.allclose(probs, [0.5, 0.0, 0.0, 0.5])


def test_psi_minus_is_anticorrelated():
    probs = probabilities(bell("psi-"))
    assert np.allclose(probs, [0.0, 0.5, 0.5, 0.0])


def test_bloch_one_qubit_states():
    # Phase 1: |0⟩ → (0, 0, 1), |+⟩ → (1, 0, 0), |+i⟩ → (0, 1, 0)
    assert np.allclose(bloch(ZERO), [0.0, 0.0, 1.0])
    assert np.allclose(bloch(PLUS), [1.0, 0.0, 0.0])
    assert np.allclose(bloch(PLUS_I), [0.0, 1.0, 0.0])

    # Pure states lie on the unit sphere S^2: r_x^2 + r_y^2 + r_z^2 = 1
    # For |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ) sin(θ/2)|1⟩:
    # r = (sin(θ)cos(φ), sin(θ)sin(φ), cos(θ))
    theta, phi = 0.7, 1.2
    psi = np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)], dtype=complex)
    r = bloch(psi)
    assert np.isclose(np.linalg.norm(r), 1.0)
    expected_r = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    assert np.allclose(r, expected_r)


def test_bloch_rejects_multiqubit():
    with pytest.raises(ValueError, match="one-qubit"):
        bloch(np.kron(ZERO, ZERO))


def test_density_and_purity():
    # Phase 2: Every current ket has purity 1
    current_kets = [ZERO, ONE, PLUS, PLUS_I, bell("phi+"), bell("phi-"), bell("psi+"), bell("psi-"), basis(2, 3)]
    for ket in current_kets:
        rho = density(ket)
        # Tr(ρ) == 1 and Tr(ρ²) == 1 for pure states
        assert np.isclose(np.trace(rho), 1.0)
        assert np.isclose(purity(rho), 1.0)
        assert np.isclose(purity(ket), 1.0)

    # The equal mixture of |0⟩ and |1⟩ has purity 1/2:
    # ρ_mix = 1/2 |0⟩⟨0| + 1/2 |1⟩⟨1| = diag(1/2, 1/2)
    # Tr(ρ_mix²) = (1/2)² + (1/2)² = 1/4 + 1/4 = 1/2
    rho_mix = 0.5 * density(ZERO) + 0.5 * density(ONE)
    assert np.isclose(purity(rho_mix), 0.5)

    # density of an unnormalized ket matches the normalized one:
    # For ket scaled by a non-zero complex factor c = (3.0 + 4.0j)
    unnormalized = (3.0 + 4.0j) * PLUS
    assert np.allclose(density(unnormalized), density(PLUS))
    assert np.isclose(purity(unnormalized), 1.0)


def test_partial_trace():
    # Phase 3:
    # 1. |00⟩ traced over qubit 1 is |0⟩⟨0| (keep qubit 0)
    ket_00 = basis(0, 2)
    rho_00 = density(ket_00)
    reduced_0 = partial_trace(rho_00, 2, keep=0)
    assert np.allclose(reduced_0, density(ZERO))
    # Also verify passing ket directly
    assert np.allclose(partial_trace(ket_00, 2, keep=0), density(ZERO))

    # 2. Either reduction of bell("phi+") is I/2
    # Traced over qubit 1 (keep 0) or traced over qubit 0 (keep 1):
    phi_plus = bell("phi+")
    red_keep_0 = partial_trace(phi_plus, 2, keep=0)
    red_keep_1 = partial_trace(phi_plus, 2, keep=1)
    i_over_2 = 0.5 * np.eye(2)
    assert np.allclose(red_keep_0, i_over_2)
    assert np.allclose(red_keep_1, i_over_2)

    # 3. Tracing both sides returns 1
    tr_both = partial_trace(phi_plus, 2, keep=[])
    assert tr_both == 1
    assert np.isclose(tr_both, 1.0)
