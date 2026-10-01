"""Pauli matrices and Kronecker products."""

from __future__ import annotations

import numpy as np

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

_PAULI = {"I": I, "X": X, "Y": Y, "Z": Z}


def pauli(label: str) -> np.ndarray:
    """Build a multi-qubit Pauli from a string such as ``XZ`` or ``YIY``."""
    if not label or any(char not in _PAULI for char in label):
        raise ValueError("label must be a non-empty string of I, X, Y, Z")
    operator = np.array([[1.0]], dtype=complex)
    for char in label:
        operator = np.kron(operator, _PAULI[char])
    return operator


def expectation(operator: np.ndarray, state: np.ndarray) -> float:
    """⟨ψ|A|ψ⟩. Imaginary part must be numerical noise."""
    ket = np.asarray(state, dtype=complex).reshape(-1)
    op = np.asarray(operator, dtype=complex)
    if op.shape != (ket.size, ket.size):
        raise ValueError("operator dimension does not match the state")
    value = np.vdot(ket, op @ ket)
    if abs(value.imag) > 1e-8:
        raise ValueError("expectation of a non-Hermitian operator")
    return float(value.real)
