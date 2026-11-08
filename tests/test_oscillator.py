import numpy as np

from eigenlab.hamiltonian import spectrum
from eigenlab.oscillator import annihilation, creation, fock, number


def test_truncated_ladder():
    # Phase 35: Truncated ladder
    for d in (4, 8, 12, 30):
        a = annihilation(d)
        a_dag = creation(d)
        num = number(d)

        # Verify action a|n> = sqrt(n) |n-1> and a|0> = 0
        assert np.allclose(a @ fock(0, d), np.zeros(d, dtype=complex), atol=1e-15)
        for n in range(1, d):
            assert np.allclose(a @ fock(n, d), np.sqrt(float(n)) * fock(n - 1, d), atol=1e-12)

        # number(d) = a†a = diag(0, ..., d-1)
        assert np.allclose(a_dag @ a, num, atol=1e-12)
        assert np.allclose(num, np.diag(np.arange(d, dtype=float)), atol=1e-15)

        # [a, a†] is I except its last diagonal entry, which is 1 - d
        comm = a @ a_dag - a_dag @ a
        expected_comm = np.eye(d, dtype=complex)
        expected_comm[-1, -1] = 1.0 - float(d)
        assert np.allclose(comm, expected_comm, atol=1e-12)

        # a†a + 1/2 has spectrum n + 1/2 exactly
        h_osc = num + 0.5 * np.eye(d, dtype=complex)
        assert np.allclose(spectrum(h_osc), np.arange(d, dtype=float) + 0.5, atol=1e-12)
