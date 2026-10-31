import numpy as np

from eigenlab.codes import bit_flip_channel
from eigenlab.pauli import X, pauli
from eigenlab.states import ONE, PLUS, ZERO, basis, density, ghz


def test_bit_flip_channel():
    # Phase 25: Independent bit flips
    # On one qubit, Kraus operators √(1−p) I and √p X give (1−p) ρ + p XρX, preserving trace.
    rho_1 = 0.5 * (np.eye(2) + 0.3 * pauli("X") - 0.4 * pauli("Y") + 0.6 * pauli("Z"))
    for p in [0.0, 0.1, 0.35, 0.5, 1.0]:
        out = bit_flip_channel(rho_1, p)
        expected = (1.0 - p) * rho_1 + p * (X @ rho_1 @ X)
        assert np.allclose(out, expected, atol=1e-12)
        assert np.isclose(np.trace(out), 1.0, atol=1e-12)

    # Multi-qubit check up to n = 8: trace preserved and product probabilities match
    for n in [2, 3, 5, 8]:
        psi = ghz(n) if n >= 2 else PLUS
        for p in [0.0, 0.15, 0.5]:
            out_n = bit_flip_channel(psi, p)
            assert out_n.shape == (1 << n, 1 << n)
            assert np.isclose(np.trace(out_n), 1.0, atol=1e-12)
            assert np.allclose(out_n, out_n.conj().T, atol=1e-12)

    # On |000⟩, population of basis state with k bit flips is p^k (1−p)^{3−k}
    p = 0.2
    out_3 = bit_flip_channel(basis(0, 3), p)
    for idx in range(8):
        k = idx.bit_count()
        assert np.isclose(np.real(out_3[idx, idx]), (p**k) * ((1.0 - p) ** (3 - k)), atol=1e-12)
