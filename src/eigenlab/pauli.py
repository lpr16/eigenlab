"""Pauli matrices and Kronecker products."""

from __future__ import annotations

import numpy as np

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

_PAULI = {"I": I, "X": X, "Y": Y, "Z": Z}

_PAULI_PRODUCT_TABLE: dict[tuple[str, str], tuple[complex, str]] = {
    ("I", "I"): (1.0 + 0.0j, "I"),
    ("I", "X"): (1.0 + 0.0j, "X"),
    ("I", "Y"): (1.0 + 0.0j, "Y"),
    ("I", "Z"): (1.0 + 0.0j, "Z"),
    ("X", "I"): (1.0 + 0.0j, "X"),
    ("X", "X"): (1.0 + 0.0j, "I"),
    ("X", "Y"): (1.0j, "Z"),
    ("X", "Z"): (-1.0j, "Y"),
    ("Y", "I"): (1.0 + 0.0j, "Y"),
    ("Y", "X"): (-1.0j, "Z"),
    ("Y", "Y"): (1.0 + 0.0j, "I"),
    ("Y", "Z"): (1.0j, "X"),
    ("Z", "I"): (1.0 + 0.0j, "Z"),
    ("Z", "X"): (1.0j, "Y"),
    ("Z", "Y"): (-1.0j, "X"),
    ("Z", "Z"): (1.0 + 0.0j, "I"),
}


def pauli_mult(a: str, b: str) -> tuple[complex, str]:
    """Multiply two Pauli strings without building matrices.

    Returns (phase, label) where phase is in {1, -1, i, -i}.
    """
    if len(a) != len(b):
        raise ValueError("labels must have the same length")
    if not a:
        raise ValueError("labels must be non-empty")
    phase = 1.0 + 0.0j
    out_chars: list[str] = []
    for c1, c2 in zip(a, b):
        if c1 not in _PAULI or c2 not in _PAULI:
            raise ValueError(f"invalid Pauli characters in '{a}' or '{b}'")
        p, c = _PAULI_PRODUCT_TABLE[(c1, c2)]
        phase *= p
        out_chars.append(c)
    return phase, "".join(out_chars)


multiply_pauli = pauli_mult
pauli_multiply = pauli_mult


def pauli_comm(a: str, b: str) -> tuple[complex, str]:
    """Commutator [A, B] = AB - BA of two equal-length Pauli strings.

    Returns (phase, label) using pauli_mult, where phase is 0 if A and B commute,
    or ±2i if they anticommute.
    """
    p1, label = pauli_mult(a, b)
    p2, _ = pauli_mult(b, a)
    return p1 - p2, label


def pauli_anti(a: str, b: str) -> tuple[complex, str]:
    """Anticommutator {A, B} = AB + BA of two equal-length Pauli strings.

    Returns (phase, label) using pauli_mult, where phase is 0 if A and B anticommute,
    or ±2 if they commute.
    """
    p1, label = pauli_mult(a, b)
    p2, _ = pauli_mult(b, a)
    return p1 + p2, label


def conjugate_pauli(target: str, conjugator: str) -> tuple[complex, str]:
    """Conjugate target Pauli string Q by conjugator P: returns (phase, label) for P Q P."""
    p1, mid = pauli_mult(conjugator, target)
    p2, out_label = pauli_mult(mid, conjugator)
    return p1 * p2, out_label


_CLIFFORD_TABLE: dict[tuple[str, str], tuple[complex, str]] = {
    ("H", "I"): (1.0 + 0.0j, "I"),
    ("H", "X"): (1.0 + 0.0j, "Z"),
    ("H", "Y"): (-1.0 + 0.0j, "Y"),
    ("H", "Z"): (1.0 + 0.0j, "X"),
    ("S", "I"): (1.0 + 0.0j, "I"),
    ("S", "X"): (1.0 + 0.0j, "Y"),
    ("S", "Y"): (-1.0 + 0.0j, "X"),
    ("S", "Z"): (1.0 + 0.0j, "Z"),
}


def conjugate_clifford(label: str, gate: str, qubit: int) -> tuple[complex, str]:
    """Conjugate a Pauli label by a single-qubit Clifford gate U in {'H', 'S'} on `qubit`: U P U†."""
    if not label or any(char not in _PAULI for char in label):
        raise ValueError("label must be a non-empty string of I, X, Y, Z")
    if gate not in ("H", "S"):
        raise ValueError(f"unsupported gate '{gate}'; must be 'H' or 'S'")
    if not 0 <= qubit < len(label):
        raise ValueError(f"qubit {qubit} out of range for label of length {len(label)}")
    chars = list(label)
    phase, new_char = _CLIFFORD_TABLE[(gate, chars[qubit])]
    chars[qubit] = new_char
    return phase, "".join(chars)


def pauli(label: str) -> np.ndarray:
    """Build a multi-qubit Pauli from a string such as ``XZ`` or ``YIY``."""
    if not label or any(char not in _PAULI for char in label):
        raise ValueError("label must be a non-empty string of I, X, Y, Z")
    operator = np.array([[1.0]], dtype=complex)
    for char in label:
        operator = np.kron(operator, _PAULI[char])
    return operator


def apply_pauli(label: str, ket: np.ndarray) -> np.ndarray:
    """Apply a Pauli string to a state ket as a signed permutation without building a 2^n x 2^n matrix."""
    if not label or any(char not in _PAULI for char in label):
        raise ValueError("label must be a non-empty string of I, X, Y, Z")
    n = len(label)
    dim = 1 << n
    vec = np.asarray(ket, dtype=complex).reshape(-1)
    if vec.size != dim:
        raise ValueError(f"ket size {vec.size} does not match 2^{n} = {dim}")

    indices = np.arange(dim, dtype=np.int64)
    parity = np.zeros(dim, dtype=np.int8)
    flip_mask = 0
    y_count = 0
    for q, ch in enumerate(label):
        bit = 1 << (n - 1 - q)
        if ch == "X":
            flip_mask |= bit
        elif ch == "Z":
            parity ^= ((indices & bit) != 0).astype(np.int8)
        elif ch == "Y":
            flip_mask |= bit
            parity ^= ((indices & bit) != 0).astype(np.int8)
            y_count += 1

    phases = (1j**y_count) * np.where(parity, -1.0, 1.0)
    out = np.empty(dim, dtype=complex)
    out[indices ^ flip_mask] = phases * vec
    return out


def expectation(operator: np.ndarray | Any, state: np.ndarray | str) -> float:
    """⟨ψ|A|ψ⟩. Imaginary part must be numerical noise."""
    if isinstance(state, str) and not isinstance(operator, str):
        operator, state = state, operator
    if isinstance(operator, str):
        ket = np.asarray(state, dtype=complex).reshape(-1)
        norm = np.linalg.norm(ket)
        if norm == 0:
            raise ValueError("state has zero norm")
        ket = ket / norm
        value = np.vdot(ket, apply_pauli(operator, ket))
        if abs(value.imag) > 1e-8:
            raise ValueError("expectation of a non-Hermitian operator")
        return float(value.real)
    if hasattr(operator, "expectation"):
        return operator.expectation(state)
    elif isinstance(operator, (list, tuple)) and operator and isinstance(operator[0], tuple):
        from eigenlab.hamiltonian import term_expectation

        return term_expectation(operator, state)
    ket = np.asarray(state, dtype=complex).reshape(-1)
    norm = np.linalg.norm(ket)
    if norm == 0:
        raise ValueError("state has zero norm")
    ket = ket / norm
    op = np.asarray(operator, dtype=complex)
    if op.shape != (ket.size, ket.size):
        raise ValueError("operator dimension does not match the state")
    value = np.vdot(ket, op @ ket)
    if abs(value.imag) > 1e-8:
        raise ValueError("expectation of a non-Hermitian operator")
    return float(value.real)
