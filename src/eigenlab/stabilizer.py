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


def _gf2_rref(M: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Compute Reduced Row Echelon Form over GF(2) and return (A, pivot_cols)."""
    A = (np.asarray(M, dtype=np.uint8) % 2).copy()
    if A.ndim == 1:
        A = A.reshape(1, -1)
    m, n = A.shape
    pivot_cols: list[int] = []
    r = 0
    for c in range(n):
        if r >= m:
            break
        pivot_rows = np.where(A[r:, c] == 1)[0]
        if len(pivot_rows) == 0:
            continue
        p_row = r + int(pivot_rows[0])
        if p_row != r:
            A[[r, p_row]] = A[[p_row, r]]
        other_rows = [i for i in range(m) if i != r and A[i, c] == 1]
        for i in other_rows:
            A[i] = A[i] ^ A[r]
        pivot_cols.append(c)
        r += 1
    return A, pivot_cols


def gf2_rank(M: np.ndarray | list) -> int:
    """Compute rank of matrix M over GF(2)."""
    A = np.asarray(M, dtype=np.uint8) % 2
    if A.ndim == 1:
        A = A.reshape(1, -1)
    if A.size == 0:
        return 0
    _, pivot_cols = _gf2_rref(A)
    return len(pivot_cols)


def gf2_nullspace(M: np.ndarray | list) -> np.ndarray:
    """Compute basis for nullspace of matrix M over GF(2).

    Returns array of shape (k, n) where each row is a basis vector v with M v = 0 (mod 2).
    """
    A = np.asarray(M, dtype=np.uint8) % 2
    if A.ndim == 1:
        A = A.reshape(1, -1)
    m, n = A.shape
    if n == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    if m == 0 or A.size == 0:
        return np.eye(n, dtype=np.uint8)

    rref, pivot_cols = _gf2_rref(A)
    pivot_set = set(pivot_cols)
    free_cols = [c for c in range(n) if c not in pivot_set]
    k = len(free_cols)

    null_basis = np.zeros((k, n), dtype=np.uint8)
    for idx, f in enumerate(free_cols):
        null_basis[idx, f] = 1
        for r, p in enumerate(pivot_cols):
            if rref[r, f] == 1:
                null_basis[idx, p] = 1
    return null_basis

