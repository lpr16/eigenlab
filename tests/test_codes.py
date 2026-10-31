import numpy as np

from eigenlab.codes import (
    BIT_FLIP_STABILIZERS,
    FIVE_QUBIT_STABILIZERS,
    PHASE_FLIP_STABILIZERS,
    bit_flip_channel,
    code_distance,
    code_projector,
    correct_bit_flip,
    correct_phase_flip,
    encode_bit_flip,
    encode_phase_flip,
    error_syndrome,
    knill_laflamme,
    phase_flip_channel,
    phase_flip_stabilizers,
    single_qubit_paulis,
    stabilizer_group,
    stabilizers_commute,
    transversal_clifford,
    weight_k_paulis,
)
from eigenlab.pauli import X, apply_pauli, conjugate_clifford, pauli, pauli_anti, pauli_comm
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


def test_phase_flip_code():
    # Phase 27: The phase-flip code
    # Conjugate the bit-flip code by H on every qubit through conjugate_clifford -> XXI and IXX.
    conjugated = []
    for s in BIT_FLIP_STABILIZERS:
        cur = s
        for q in range(3):
            phase, cur = conjugate_clifford(cur, "H", q)
            assert phase == 1.0
        conjugated.append(cur)
    assert tuple(conjugated) == ("XXI", "IXX")
    assert PHASE_FLIP_STABILIZERS == ("XXI", "IXX")
    assert phase_flip_stabilizers() == ("XXI", "IXX")
    assert stabilizers_commute(PHASE_FLIP_STABILIZERS)

    # Corrects single Z errors with the same 3p² − 2p³ failure probability
    errors_z = ["III", "ZII", "IZI", "IIZ"]
    syndromes_z = [error_syndrome(e, PHASE_FLIP_STABILIZERS) for e in errors_z]
    assert syndromes_z == [(1, 1), (-1, 1), (-1, -1), (1, -1)]

    psi_1q = np.array([0.6, 0.8], dtype=complex)
    psi_L = encode_phase_flip(psi_1q)
    flipped_L = apply_pauli("ZZZ", psi_L)
    rho_good = density(psi_L)
    rho_bad = density(flipped_L)

    for p in [0.0, 0.1, 0.5]:
        p_fail = 3.0 * (p**2) - 2.0 * (p**3)
        noisy = phase_flip_channel(psi_L, p)
        corrected = correct_phase_flip(noisy)
        expected = (1.0 - p_fail) * rho_good + p_fail * rho_bad
        assert np.allclose(corrected, expected, atol=1e-12)

    # The bit-flip code does not correct Z errors:
    # ZII commutes with ZZI and IZZ, is not in their stabilizer group, and flips the logical phase.
    assert pauli_comm("ZII", "ZZI")[0] == 0
    assert pauli_comm("ZII", "IZZ")[0] == 0
    bf_group = stabilizer_group(BIT_FLIP_STABILIZERS)
    assert bf_group == {"III", "ZZI", "IZZ", "ZIZ"}
    assert "ZII" not in bf_group

    alpha, beta = 0.6, 0.8
    bf_psi = encode_bit_flip(np.array([alpha, beta]))
    bf_phase_flipped = encode_bit_flip(np.array([alpha, -beta]))
    assert np.allclose(apply_pauli("ZII", bf_psi), bf_phase_flipped, atol=1e-12)
    assert np.allclose(correct_bit_flip(apply_pauli("ZII", bf_psi)), density(bf_phase_flipped), atol=1e-12)


def test_five_qubit_code():
    # Phase 28: The five-qubit code
    # Generators XZZXI, IXZZX, XIXZZ, ZXIXZ commute pairwise, and the code space has rank 2.
    assert FIVE_QUBIT_STABILIZERS == ("XZZXI", "IXZZX", "XIXZZ", "ZXIXZ")
    assert stabilizers_commute(FIVE_QUBIT_STABILIZERS)

    proj = code_projector(FIVE_QUBIT_STABILIZERS)
    assert np.allclose(proj @ proj, proj, atol=1e-12)
    assert np.isclose(np.real(np.trace(proj)), 2.0, atol=1e-12)
    assert np.linalg.matrix_rank(proj, tol=1e-10) == 2

    # Identity + 15 single-qubit Paulis give 16 distinct syndromes (all 2^4 of them).
    paulis_1q = single_qubit_paulis(5)
    assert len(paulis_1q) == 15
    errors = ["IIIII"] + paulis_1q
    syndromes = [error_syndrome(e, FIVE_QUBIT_STABILIZERS) for e in errors]
    assert len(syndromes) == 16
    assert len(set(syndromes)) == 16


def test_five_qubit_code_distance_three():
    # Phase 29: Distance three
    # No Pauli of weight 1 or 2 commutes with all four generators (15 + 90 = 105 labels).
    w1 = weight_k_paulis(5, 1)
    w2 = weight_k_paulis(5, 2)
    assert len(w1) == 15
    assert len(w2) == 90
    w1_w2 = w1 + w2
    assert len(w1_w2) == 105
    for lbl in w1_w2:
        assert error_syndrome(lbl, FIVE_QUBIT_STABILIZERS) != (1, 1, 1, 1)

    group = stabilizer_group(FIVE_QUBIT_STABILIZERS)
    assert len(group) == 16

    # XXXXX and ZZZZZ commute with every generator, are not in the stabilizer group,
    # and anticommute with each other.
    for log_op in ["XXXXX", "ZZZZZ"]:
        assert error_syndrome(log_op, FIVE_QUBIT_STABILIZERS) == (1, 1, 1, 1)
        assert log_op not in group
    assert pauli_comm("XXXXX", "ZZZZZ")[0] != 0
    assert pauli_anti("XXXXX", "ZZZZZ")[0] == 0

    # Weight-3 Pauli that commutes with every generator without being in the group:
    # XXXXX * XZZXI = IYYIX has weight 3 and lies in the logical X coset.
    w3_logicals = [
        lbl
        for lbl in weight_k_paulis(5, 3)
        if error_syndrome(lbl, FIVE_QUBIT_STABILIZERS) == (1, 1, 1, 1) and lbl not in group
    ]
    assert "IYYIX" in w3_logicals
    assert len(w3_logicals) > 0
    assert code_distance(FIVE_QUBIT_STABILIZERS) == 3


def test_knill_laflamme():
    # Phase 30: Knill-Laflamme
    # For the five-qubit code and E over I and the 15 single-qubit Paulis,
    # P E_a† E_b P = c_ab P for all 256 pairs, with c_ab = Tr(P E_a† E_b P) / 2.
    P_5 = code_projector(FIVE_QUBIT_STABILIZERS)
    errors_5 = ["IIIII"] + single_qubit_paulis(5)
    assert len(errors_5) == 16

    ok_5, c_5 = knill_laflamme(P_5, errors_5)
    assert ok_5
    # Because the 16 syndromes are distinct, P E_a† E_b P = δ_ab P (degenerate/orthogonal syndromes)
    assert np.allclose(c_5, np.eye(16), atol=1e-12)

    # For the bit-flip code with E over I and single-qubit X and Z, the condition fails,
    # e.g. on the pair ("III", "ZII") where P ZII P = |000⟩⟨000| − |111⟩⟨111| != c P (c = 0).
    P_bf = code_projector(BIT_FLIP_STABILIZERS)
    errors_bf = ["III", "XII", "IXI", "IIX", "ZII", "IZI", "IIZ"]
    ok_bf, _ = knill_laflamme(P_bf, errors_bf)
    assert not ok_bf

    # Named failing pair: E_a = "III", E_b = "ZII"
    e_a = pauli("III")
    e_b = pauli("ZII")
    block_fail = P_bf @ e_a.conj().T @ e_b @ P_bf
    c_fail = np.trace(block_fail) / 2.0
    assert np.isclose(c_fail, 0.0, atol=1e-12)
    assert not np.allclose(block_fail, c_fail * P_bf, atol=1e-6)





