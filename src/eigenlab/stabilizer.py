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


def z2_symmetries(hamiltonian: Hamiltonian) -> list[str]:
    """Find independent Z_2 symmetry generators (made of I and Z) of a Hamiltonian.

    A Z-type string with bit vector a commutes with a term with x-part x_t
    iff a · x_t = 0 (mod 2). The symmetries are therefore the GF(2) nullspace
    of the matrix formed by the terms' x-parts. The identity is excluded.
    """
    from eigenlab.hamiltonian import Hamiltonian

    n = hamiltonian.qubits
    if n == 0:
        return []
    terms = [(c, l) for c, l in hamiltonian.terms if abs(c) > 1e-12]
    if not terms:
        return ["I" * i + "Z" + "I" * (n - 1 - i) for i in range(n)]

    x_rows = [pauli_to_bits(lbl)[0] for _, lbl in terms]
    mx = np.array(x_rows, dtype=np.uint8)
    ns = gf2_nullspace(mx)
    if ns.size == 0 or len(ns) == 0:
        return []

    rref, _ = _gf2_rref(ns)
    nonzero_rows = [row for row in rref if np.any(row)]

    zero_x = np.zeros(n, dtype=np.uint8)
    labels = [
        bits_to_pauli(zero_x, row)
        for row in nonzero_rows
        if bits_to_pauli(zero_x, row) != "I" * n
    ]
    return labels


def z2_taper(hamiltonian, signs, generators=None):
    """Taper a Hamiltonian on its Z2 symmetries using Clifford transformations."""
    from eigenlab.hamiltonian import z2_taper as _z2_taper

    return _z2_taper(hamiltonian, signs, generators)


class SignedPauli(tuple):
    """A signed Pauli operator: (sign, label)."""

    def __new__(cls, sign: int, label: str):
        return super().__new__(cls, (int(sign), str(label)))

    @property
    def sign(self) -> int:
        return self[0]

    @property
    def label(self) -> str:
        return self[1]

    def __str__(self) -> str:
        s = "+" if self.sign > 0 else "-"
        return f"{s}{self.label}"

    def __repr__(self) -> str:
        s = "+" if self.sign > 0 else "-"
        return f"SignedPauli('{s}{self.label}')"

    def __eq__(self, other: object) -> bool:
        if (
            isinstance(other, tuple)
            and len(other) == 2
            and isinstance(other[0], int)
            and isinstance(other[1], str)
        ):
            return self[0] == other[0] and self[1] == other[1]
        if isinstance(other, str):
            return str(self) == other
        return False


class Tableau:
    """Stabilizer tableau for n qubits following Aaronson and Gottesman (2004)."""

    def __init__(self, n: int):
        if n < 1:
            raise ValueError("n must be at least 1")
        self.n = n
        # Rows 0 ... n-1: destabilizers
        # Rows n ... 2n-1: stabilizers
        self.x = np.zeros((2 * n, n), dtype=np.uint8)
        self.z = np.zeros((2 * n, n), dtype=np.uint8)
        self.r = np.zeros(2 * n, dtype=np.uint8)
        for i in range(n):
            self.x[i, i] = 1
            self.z[n + i, i] = 1

    def copy(self) -> Tableau:
        other = Tableau(self.n)
        other.x = self.x.copy()
        other.z = self.z.copy()
        other.r = self.r.copy()
        return other

    def h(self, q: int) -> Tableau:
        """Apply Hadamard gate on qubit q."""
        if not 0 <= q < self.n:
            raise ValueError(f"qubit {q} out of range for {self.n} qubits")
        self.r ^= self.x[:, q] & self.z[:, q]
        self.x[:, q], self.z[:, q] = self.z[:, q].copy(), self.x[:, q].copy()
        return self

    def s(self, q: int) -> Tableau:
        """Apply Phase gate S on qubit q."""
        if not 0 <= q < self.n:
            raise ValueError(f"qubit {q} out of range for {self.n} qubits")
        self.r ^= self.x[:, q] & self.z[:, q]
        self.z[:, q] ^= self.x[:, q]
        return self

    def cnot(self, c: int, t: int) -> Tableau:
        """Apply CNOT gate with control c and target t."""
        if not (0 <= c < self.n and 0 <= t < self.n):
            raise ValueError(f"qubits ({c}, {t}) out of range for {self.n} qubits")
        if c == t:
            raise ValueError("control and target must be distinct")
        self.r ^= self.x[:, c] & self.z[:, t] & (self.x[:, t] ^ self.z[:, c] ^ 1)
        self.x[:, t] ^= self.x[:, c]
        self.z[:, c] ^= self.z[:, t]
        return self

    cx = cnot

    def stabilizers(self) -> list[SignedPauli]:
        """Return the n stabilizer generators as signed labels."""
        res = []
        for i in range(self.n, 2 * self.n):
            sign = -1 if self.r[i] else +1
            lbl = bits_to_pauli(self.x[i], self.z[i])
            res.append(SignedPauli(sign, lbl))
        return res

    def destabilizers(self) -> list[SignedPauli]:
        """Return the n destabilizer generators as signed labels."""
        res = []
        for i in range(self.n):
            sign = -1 if self.r[i] else +1
            lbl = bits_to_pauli(self.x[i], self.z[i])
            res.append(SignedPauli(sign, lbl))
        return res




