import numpy as np

from eigenlab.codes import (
    BIT_FLIP_STABILIZERS,
    bit_flip_channel,
    code_projector,
    correct_bit_flip,
    encode_bit_flip,
    error_syndrome,
    stabilizers_commute,
)
from eigenlab.pauli import X, apply_pauli, pauli, pauli_comm
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


def test_bit_flip_code():
    # Phase 26: The bit-flip code
    # Stabilizers ZZI and IZZ commute, and Π (I + S)/2 has rank 2.
    assert BIT_FLIP_STABILIZERS == ("ZZI", "IZZ")
    assert pauli_comm("ZZI", "IZZ")[0] == 0
    assert stabilizers_commute(BIT_FLIP_STABILIZERS)

    proj = code_projector(BIT_FLIP_STABILIZERS)
    assert np.allclose(proj @ proj, proj, atol=1e-12)
    assert np.isclose(np.real(np.trace(proj)), 2.0, atol=1e-12)
    assert np.linalg.matrix_rank(proj, tol=1e-10) == 2

    # The four syndromes distinguish III, XII, IXI, and IIX
    errors = ["III", "XII", "IXI", "IIX"]
    syndromes = [error_syndrome(e, BIT_FLIP_STABILIZERS) for e in errors]
    assert syndromes == [(1, 1), (-1, 1), (-1, -1), (1, -1)]
    assert len(set(syndromes)) == 4

    # Encode α|000⟩ + β|111⟩, apply bit_flip_channel, measure syndrome, and correct by majority.
    # Logical failure probability is 3p²(1−p) + p³ = 3p² − 2p³.
    psi_1q = np.array([0.6, 0.8j], dtype=complex)
    psi_L = encode_bit_flip(psi_1q)
    flipped_L = apply_pauli("XXX", psi_L)
    rho_good = density(psi_L)
    rho_bad = density(flipped_L)

    for p in [0.0, 0.1, 0.5]:
        p_fail = 3.0 * (p**2) * (1.0 - p) + (p**3)
        assert np.isclose(p_fail, 3.0 * (p**2) - 2.0 * (p**3), atol=1e-15)

        noisy = bit_flip_channel(psi_L, p)
        corrected = correct_bit_flip(noisy)
        expected_corrected = (1.0 - p_fail) * rho_good + p_fail * rho_bad
        assert np.allclose(corrected, expected_corrected, atol=1e-12)

        # Direct check on |000⟩ where ⟨111|ρ_corr|111⟩ = p_fail
        corr_zero = correct_bit_flip(bit_flip_channel(basis(0, 3), p))
        assert np.isclose(1.0 - np.real(corr_zero[0, 0]), p_fail, atol=1e-12)
        assert np.isclose(np.real(corr_zero[0b111, 0b111]), p_fail, atol=1e-12)

        if 0.0 < p < 0.5:
            assert p_fail < p
        if np.isclose(p, 0.5):
            assert np.isclose(p_fail, 0.5, atol=1e-15)

