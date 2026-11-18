"""Bell operators: Mermin, Peres–Mermin, and the Horodecki CHSH criterion."""

import itertools

import numpy as np

from eigenlab.nonlocality import mermin_three
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
