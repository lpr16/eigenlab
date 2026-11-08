import numpy as np

from eigenlab.hamiltonian import spectrum
from eigenlab.oscillator import (
    annihilation,
    coherent,
    creation,
    excitation_number,
    fock,
    jaynes_cummings,
    number,
)


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


def test_coherent_states():
    # Phase 36: Coherent states
    import math

    for alpha in (0.7, 1.2 + 0.5j, -1.5j):
        for d in (6, 12, 20):
            c = coherent(alpha, d)
            norm_sq = float(np.vdot(c, c).real)
            lam = abs(alpha) ** 2
            poisson_sum = sum(math.exp(-lam) * (lam**n) / math.factorial(n) for n in range(d))
            assert np.isclose(norm_sq, poisson_sum, atol=1e-12)

            # a c = alpha c - alpha c_{d-1} |d-1>
            a = annihilation(d)
            rhs = alpha * c - alpha * c[d - 1] * fock(d - 1, d)
            assert np.allclose(a @ c, rhs, atol=1e-12)

    # At alpha = 2, d = 30, 1 - ||c||^2 is below 1e-12
    c_30 = coherent(2.0, 30)
    norm_sq_30 = float(np.vdot(c_30, c_30).real)
    assert abs(1.0 - norm_sq_30) < 1e-12
    assert 1.0 - norm_sq_30 < 1e-12


def test_jaynes_cummings():
    # Phase 37: Jaynes-Cummings
    from eigenlab.states import ONE, ZERO

    d = 12
    omega_c = 1.0
    omega_q = 1.15
    g = 0.05

    H = jaynes_cummings(d, omega_c=omega_c, omega_q=omega_q, g=g)
    N_tot = excitation_number(d)

    # Commutes with a†a ⊗ I + I ⊗ |1><1|
    assert np.allclose(H @ N_tot - N_tot @ H, np.zeros_like(H), atol=1e-12)

    # Ground state is |0>|0> at -omega_q / 2
    gs = np.kron(fock(0, d), ZERO)
    assert np.allclose(H @ gs, (-0.5 * omega_q) * gs, atol=1e-12)

    # Top unpartnered state |d-1>|1> sits alone at (d-1) * omega_c + omega_q / 2
    top = np.kron(fock(d - 1, d), ONE)
    assert np.allclose(H @ top, ((d - 1) * omega_c + 0.5 * omega_q) * top, atol=1e-12)

    # Full spectrum of 2d values
    delta = omega_q - omega_c
    expected: list[float] = [-0.5 * omega_q, (d - 1) * omega_c + 0.5 * omega_q]
    for n in range(d - 1):
        mid = (n + 0.5) * omega_c
        split = 0.5 * np.sqrt(delta**2 + 4.0 * (g**2) * (n + 1))
        expected.append(float(mid - split))
        expected.append(float(mid + split))

    expected_sorted = np.sort(np.array(expected, dtype=float))
    assert expected_sorted.size == 2 * d
    assert np.allclose(spectrum(H), expected_sorted, atol=1e-12)


