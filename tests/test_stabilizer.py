import itertools
import numpy as np
import pytest

from eigenlab.pauli import pauli_comm, pauli_mult
from eigenlab.stabilizer import bits_to_pauli, pauli_to_bits


def test_paulis_as_bit_vectors():
    # Phase 1: Paulis as bit vectors
    # Single qubit check
    assert bits_to_pauli([0], [0]) == "I"
    assert bits_to_pauli([1], [0]) == "X"
    assert bits_to_pauli([0], [1]) == "Z"
    assert bits_to_pauli([1], [1]) == "Y"

    # Round trips are exact
    for length in [1, 2, 3]:
        for p in itertools.product("IXYZ", repeat=length):
            label = "".join(p)
            x, z = pauli_to_bits(label)
            assert x.dtype == np.uint8
            assert z.dtype == np.uint8
            assert bits_to_pauli(x, z) == label

    # Commutation relation: x1·z2 + z1·x2 = 0 (mod 2)
    # Check length-2 labels (256 pairs)
    labels_2 = ["".join(p) for p in itertools.product("IXYZ", repeat=2)]
    for a in labels_2:
        x1, z1 = pauli_to_bits(a)
        for b in labels_2:
            x2, z2 = pauli_to_bits(b)
            symplectic_inner = int((np.dot(x1, z2) + np.dot(z1, x2)) % 2)
            comm_phase, _ = pauli_comm(a, b)
            is_commuting = comm_phase == 0.0
            assert (symplectic_inner == 0) == is_commuting

    # Check length-3 labels (4096 pairs)
    labels_3 = ["".join(p) for p in itertools.product("IXYZ", repeat=3)]
    # All 64 x 64 = 4096 pairs
    for a in labels_3:
        x1, z1 = pauli_to_bits(a)
        for b in labels_3:
            x2, z2 = pauli_to_bits(b)
            symplectic_inner = int((np.dot(x1, z2) + np.dot(z1, x2)) % 2)
            comm_phase, _ = pauli_comm(a, b)
            is_commuting = comm_phase == 0.0
            assert (symplectic_inner == 0) == is_commuting

    # Multiplying two Paulis adds their bit vectors mod 2
    # For every pair of length-2 labels, bits_to_pauli of the sum is the label pauli_mult returns
    for a in labels_2:
        x1, z1 = pauli_to_bits(a)
        for b in labels_2:
            x2, z2 = pauli_to_bits(b)
            sum_x = (x1 + x2) % 2
            sum_z = (z1 + z2) % 2
            mult_label = bits_to_pauli(sum_x, sum_z)
            _, expected_label = pauli_mult(a, b)
            assert mult_label == expected_label
