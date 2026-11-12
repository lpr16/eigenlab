import numpy as np
import pytest

from eigenlab.clifford import one_qubit_cliffords
from eigenlab.pauli import X, Y, Z


def test_twenty_four_elements():
    # Phase 13: Twenty-four elements
    # The one-qubit Clifford group modulo global phase is isomorphic to the octahedral group O ~= S_4,
    # having order 24.
    # The map C ↦ (C X C†, C Z C†) is injective because:
    # 1. X can be mapped to any of the 6 signed single-qubit Paulis: {+X, -X, +Y, -Y, +Z, -Z}.
    # 2. Since {X, Z} = 0, C Z C† must anticommute with C X C†.
    #    Out of the 6 signed Paulis, exactly 2 commute with C X C† (itself and its negative),
    #    leaving exactly 4 signed Paulis that anticommute with C X C†.
    # Thus there are 6 * 4 = 24 possible pairs (C X C†, C Z C†), and because the action of C
    # on X and Z completely determines its action on Y = i X Z and on the entire matrix modulo global phase,
    # the mapping is injective and bijection holds.
    cliffords = one_qubit_cliffords()

    # 1. There are 24 elements
    assert len(cliffords) == 24

    signed_paulis = {
        "+X": X,
        "-X": -X,
        "+Y": Y,
        "-Y": -Y,
        "+Z": Z,
        "-Z": -Z,
    }

    def identify_signed_pauli(mat: np.ndarray) -> str:
        for name, p in signed_paulis.items():
            if np.allclose(mat, p, atol=1e-8):
                return name
        raise AssertionError(f"Matrix {mat} is not a signed Pauli")

    pairs = set()
    for c in cliffords:
        # Phase check: first nonzero entry is real and positive
        for val in c.ravel():
            if abs(val) > 1e-10:
                assert val.real > 0
                assert np.isclose(val.imag, 0.0, atol=1e-10)
                break

        # 2. Each conjugates X, Y and Z to signed Paulis
        cx = identify_signed_pauli(c @ X @ c.conj().T)
        cy = identify_signed_pauli(c @ Y @ c.conj().T)
        cz = identify_signed_pauli(c @ Z @ c.conj().T)

        # 3. Pair (C X C†, C Z C†)
        pairs.add((cx, cz))

    # The map C ↦ (C X C†, C Z C†) is injective: all 24 pairs are distinct
    assert len(pairs) == 24
