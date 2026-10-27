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
