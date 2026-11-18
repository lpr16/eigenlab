"""Bell operators: Mermin, Peres–Mermin, and the Horodecki CHSH criterion."""

import itertools

import numpy as np

from eigenlab.nonlocality import mermin_operator, mermin_three
from eigenlab.states import ghz


def test_mermin_three():
    # Phase 25: Mermin, three qubits
    # M = XXX − XYY − YXY − YYX.
    # On GHZ, ⟨M⟩ = 4, which is also the largest eigenvalue.
    # A local deterministic assignment gives each of x_k, y_k a sign ±1.
    # The largest of the 4^3 = 64 assignments is 2.
    m = mermin_three()
    state = ghz(3)
    assert np.isclose(m.expectation(state), 4.0, atol=1e-12)
    energies = np.linalg.eigvalsh(m.to_matrix())
    assert np.isclose(energies[-1], 4.0, atol=1e-12)

    local_max = -np.inf
    for bits in itertools.product((-1.0, 1.0), repeat=6):
        x0, x1, x2, y0, y1, y2 = bits
        value = (
            x0 * x1 * x2
            - x0 * y1 * y2
            - y0 * x1 * y2
            - y0 * y1 * x2
        )
        local_max = max(local_max, value)
    assert local_max == 2.0


def test_mermin_enumeration():
    # Phase 26: Mermin by enumeration
    # ∏_k (X_k + i Y_k) = 2^n |0…0⟩⟨1…1|, so the real part has eigenvalues
    # ±2^{n−1} and 0, and ⟨ghz|M|ghz⟩ = 2^{n−1}.
    # Each local factor x + i y has modulus √2 and a phase that is an odd
    # multiple of π/4. The largest of the 4^n assignments is 2^{n/2} (even n)
    # or 2^{(n−1)/2} (odd n).
    assert np.allclose(mermin_operator(3).to_matrix(), mermin_three().to_matrix(), atol=1e-12)

    for x, y in itertools.product((-1.0, 1.0), repeat=2):
        factor = x + 1j * y
        assert np.isclose(abs(factor), np.sqrt(2.0))
        phase = np.angle(factor) / (np.pi / 4.0)
        assert np.isclose(phase, round(phase))
        assert int(round(phase)) % 2 == 1

    ratios = []
    for n in range(3, 9):
        m = mermin_operator(n)
        assert np.isclose(m.expectation(ghz(n)), 2.0 ** (n - 1), atol=1e-10)
        energies = np.linalg.eigvalsh(m.to_matrix())
        nonzero = energies[np.abs(energies) > 1e-8]
        assert len(nonzero) == 2
        assert np.allclose(sorted(nonzero), [-(2.0 ** (n - 1)), 2.0 ** (n - 1)], atol=1e-8)

        local_max = 0.0
        for signs in itertools.product((-1.0, 1.0), repeat=2 * n):
            value = 1.0 + 0.0j
            for k in range(n):
                value *= signs[k] + 1j * signs[n + k]
            local_max = max(local_max, float(np.real(value)))
        if n % 2 == 0:
            expected_local = 2.0 ** (n / 2)
        else:
            expected_local = 2.0 ** ((n - 1) / 2)
        assert np.isclose(local_max, expected_local, atol=1e-10)
        ratios.append(2.0 ** (n - 1) / expected_local)

    assert ratios[-1] > ratios[0]
    assert all(b >= a for a, b in zip(ratios, ratios[1:]))
