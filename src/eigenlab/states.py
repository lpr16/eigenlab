"""Computational-basis kets and two-qubit Bell states."""

from __future__ import annotations

import numpy as np

ZERO = np.array([1, 0], dtype=complex)
ONE = np.array([0, 1], dtype=complex)
PLUS = np.array([1, 1], dtype=complex) / np.sqrt(2)


def basis(index: int, qubits: int) -> np.ndarray:
    """Computational-basis ket |index⟩ on `qubits` qubits."""
    if qubits < 1 or not 0 <= index < (1 << qubits):
        raise ValueError("index out of range for the given number of qubits")
    ket = np.zeros(1 << qubits, dtype=complex)
    ket[index] = 1
    return ket


def bell(kind: str = "phi+") -> np.ndarray:
    """One of the four Bell states."""
    phi_plus = (np.kron(ZERO, ZERO) + np.kron(ONE, ONE)) / np.sqrt(2)
    phi_minus = (np.kron(ZERO, ZERO) - np.kron(ONE, ONE)) / np.sqrt(2)
    psi_plus = (np.kron(ZERO, ONE) + np.kron(ONE, ZERO)) / np.sqrt(2)
    psi_minus = (np.kron(ZERO, ONE) - np.kron(ONE, ZERO)) / np.sqrt(2)
    states = {"phi+": phi_plus, "phi-": phi_minus, "psi+": psi_plus, "psi-": psi_minus}
    try:
        return states[kind]
    except KeyError as exc:
        raise ValueError("kind must be phi+, phi-, psi+, or psi-") from exc


def probabilities(state: np.ndarray) -> np.ndarray:
    ket = np.asarray(state, dtype=complex).reshape(-1)
    norm = np.linalg.norm(ket)
    if norm == 0:
        raise ValueError("state has zero norm")
    return np.abs(ket / norm) ** 2
