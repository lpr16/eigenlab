"""Stabilizer formalisms, GF(2) linear algebra, and tableaux."""

from __future__ import annotations

import numpy as np

_CHAR_TO_BITS: dict[str, tuple[int, int]] = {
    "I": (0, 0),
    "X": (1, 0),
    "Z": (0, 1),
    "Y": (1, 1),
}

_BITS_TO_CHAR: dict[tuple[int, int], str] = {
    (0, 0): "I",
    (1, 0): "X",
    (0, 1): "Z",
    (1, 1): "Y",
}


def pauli_to_bits(label: str) -> tuple[np.ndarray, np.ndarray]:
    """Convert a Pauli string label to bit vectors (x, z) as uint8 arrays."""
    x = []
    z = []
    for c in label:
        if c not in _CHAR_TO_BITS:
            raise ValueError(f"invalid Pauli character '{c}' in label '{label}'")
        bx, bz = _CHAR_TO_BITS[c]
        x.append(bx)
        z.append(bz)
    return np.array(x, dtype=np.uint8), np.array(z, dtype=np.uint8)


def bits_to_pauli(x: np.ndarray | list[int], z: np.ndarray | list[int]) -> str:
    """Convert bit vectors (x, z) to a Pauli string label."""
    xa = np.asarray(x, dtype=np.uint8).ravel()
    za = np.asarray(z, dtype=np.uint8).ravel()
    if len(xa) != len(za):
        raise ValueError(f"x and z must have same length, got {len(xa)} and {len(za)}")
    chars = []
    for bx, bz in zip(xa, za):
        chars.append(_BITS_TO_CHAR[(int(bx % 2), int(bz % 2))])
    return "".join(chars)
