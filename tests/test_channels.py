import numpy as np

from eigenlab.states import ONE, ZERO, density, purity


def test_amplitude_damping_kraus():
    # Phase 39: Amplitude-damping Kraus map
    # E0† E0 = diag(1, 1−p) and E1† E1 = diag(0, p), so the sum is I.
    # On |1⟩⟨1|: E0 |1⟩⟨1| E0† = (1−p)|1⟩⟨1| and E1 |1⟩⟨1| E1† = p|0⟩⟨0|.
    from eigenlab.channels import amplitude_damping, amplitude_damping_kraus

    excited = density(ONE)
    for p in [0.0, 0.25, 0.5, 0.9, 1.0]:
        e0, e1 = amplitude_damping_kraus(p)
        assert np.allclose(e0.conj().T @ e0 + e1.conj().T @ e1, np.eye(2), atol=1e-12)

        out = amplitude_damping(excited, p)
        assert np.allclose(np.real(np.diag(out)), [p, 1.0 - p], atol=1e-12)
        assert np.isclose(purity(out), p**2 + (1.0 - p) ** 2, atol=1e-12)

    assert np.allclose(amplitude_damping(excited, 0.0), excited, atol=1e-12)
    assert np.allclose(amplitude_damping(excited, 1.0), density(ZERO), atol=1e-12)


def test_depolarize():
    # Phase 40: Depolarizing channel
    # Conjugation by X keeps r_x and flips r_y, r_z (likewise for Y and Z), so
    # XρX + YρY + ZρZ has Bloch vector −r and trace 3: the channel sends r → (1 − p − p/3) r.
    # On |0⟩⟨0|, r_z = 1 → 1 − 4p/3, so the populations are (1 − 2p/3, 2p/3).
    from eigenlab.channels import depolarize
    from eigenlab.states import bloch

    ground = density(ZERO)
    for p in [0.0, 0.1, 0.5, 0.75, 1.0]:
        out = depolarize(ground, p)
        assert np.isclose(np.trace(out), 1.0, atol=1e-12)
        assert np.allclose(np.real(np.diag(out)), [1.0 - 2.0 * p / 3.0, 2.0 * p / 3.0], atol=1e-12)

    theta, phi = 0.7, 1.2
    psi = np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)], dtype=complex)
    rho = density(psi)
    assert np.allclose(bloch(rho), bloch(psi), atol=1e-12)
    for p in [0.0, 0.2, 0.5, 0.75, 0.9]:
        assert np.allclose(bloch(depolarize(rho, p)), (1.0 - 4.0 * p / 3.0) * bloch(psi), atol=1e-12)

    assert np.allclose(depolarize(rho, 0.0), rho, atol=1e-12)
    assert np.allclose(depolarize(rho, 0.75), np.eye(2) / 2.0, atol=1e-12)


def _mixed_qubit() -> np.ndarray:
    # A mixed state with populations and a complex coherence: r = (0.3, −0.4, 0.5), |r| < 1.
    from eigenlab.pauli import X, Y, Z

    return 0.5 * (np.eye(2) + 0.3 * X - 0.4 * Y + 0.5 * Z)


def test_dephasing_closed_form():
    # Phase 41: Dephasing, closed form
    # (ZρZ)₀₀ = ρ₀₀, (ZρZ)₁₁ = ρ₁₁, (ZρZ)₀₁ = −ρ₀₁, so
    #   dρ₀₀/dt = dρ₁₁/dt = 0 and dρ₀₁/dt = −2γ ρ₀₁, i.e. ρ₀₁(t) = ρ₀₁(0) e^{−2γt}.
    # |+⟩⟨+| has ρ₀₁(0) = 1/2. At γ = 1:
    #   t = ln 2:     e^{−2 ln 2} = 1/4, so ρ₀₁ = 1/8.
    #   t = (ln 2)/2: e^{−ln 2} = 1/2,  so ρ₀₁ = 1/4.
    from eigenlab.channels import dephasing
    from eigenlab.pauli import Z
    from eigenlab.states import PLUS

    plus = density(PLUS)
    assert np.isclose(dephasing(plus, 1.0, 0.0)[0, 1], 0.5, atol=1e-12)
    assert np.isclose(dephasing(plus, 1.0, np.log(2.0))[0, 1], 0.125, atol=1e-12)
    assert np.isclose(dephasing(plus, 1.0, np.log(2.0) / 2.0)[0, 1], 0.25, atol=1e-12)
    assert np.allclose(np.diag(dephasing(plus, 1.0, np.log(2.0))), [0.5, 0.5], atol=1e-12)

    # The implemented map solves the master equation.
    gamma = 0.8
    rho0 = _mixed_qubit()
    step = 1e-6
    for t in [0.0, 0.3, np.log(2.0)]:
        central = (dephasing(rho0, gamma, t + step) - dephasing(rho0, gamma, t - step)) / (2.0 * step)
        rho_t = dephasing(rho0, gamma, t)
        rhs = gamma * (Z @ rho_t @ Z - rho_t)
        assert np.max(np.abs(central - rhs)) < 1e-8


def test_amplitude_damping_closed_form():
    # Phase 42: Amplitude damping, closed form
    # σ₊σ₋ = |1⟩⟨1| and σ₋ρσ₊ = ρ₁₁ |0⟩⟨0|, so
    #   dρ₁₁/dt = −γ ρ₁₁                  → ρ₁₁(t) = ρ₁₁(0) e^{−γt}
    #   dρ₀₁/dt = −(γ/2) ρ₀₁              → ρ₀₁(t) = ρ₀₁(0) e^{−γt/2}
    #   dρ₀₀/dt = +γ ρ₁₁                  → ρ₀₀(t) = ρ₀₀(0) + ρ₁₁(0) (1 − e^{−γt})
    # The Kraus map of phase 39 gives (1−p) ρ₁₁ and √(1−p) ρ₀₁, the same with p = 1 − e^{−γt}.
    from eigenlab.channels import amplitude_damping, amplitude_damping_evolve

    sigma_minus = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    sigma_plus = sigma_minus.conj().T

    # On |1⟩⟨1| at γ = 1, t = ln 2: e^{−ln 2} = 1/2.
    excited = density(ONE)
    out = amplitude_damping_evolve(excited, 1.0, np.log(2.0))
    assert np.isclose(out[1, 1], 0.5, atol=1e-12)
    assert np.isclose(out[0, 0], 0.5, atol=1e-12)

    gamma = 0.8
    rho0 = _mixed_qubit()
    step = 1e-6
    for t in [0.0, 0.3, np.log(2.0)]:
        central = (
            amplitude_damping_evolve(rho0, gamma, t + step) - amplitude_damping_evolve(rho0, gamma, t - step)
        ) / (2.0 * step)
        rho_t = amplitude_damping_evolve(rho0, gamma, t)
        n_op = sigma_plus @ sigma_minus
        rhs = gamma * (sigma_minus @ rho_t @ sigma_plus - 0.5 * (n_op @ rho_t + rho_t @ n_op))
        assert np.max(np.abs(central - rhs)) < 1e-8

        p = 1.0 - np.exp(-gamma * t)
        assert np.allclose(rho_t, amplitude_damping(rho0, p), atol=1e-12)
