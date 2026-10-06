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


def pauli(label: str) -> np.ndarray:
    """Build a multi-qubit Pauli from a string such as ``XZ`` or ``YIY``."""
    if not label or any(char not in _PAULI for char in label):
        raise ValueError("label must be a non-empty string of I, X, Y, Z")
    operator = np.array([[1.0]], dtype=complex)
    for char in label:
        operator = np.kron(operator, _PAULI[char])
    return operator


def expectation(operator: np.ndarray | Any, state: np.ndarray) -> float:
    """⟨ψ|A|ψ⟩. Imaginary part must be numerical noise."""
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
