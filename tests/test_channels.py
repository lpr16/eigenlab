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
