"""Computational-basis kets and two-qubit Bell states."""

from __future__ import annotations

import numpy as np

ZERO = np.array([1, 0], dtype=complex)
ONE = np.array([0, 1], dtype=complex)
PLUS = np.array([1, 1], dtype=complex) / np.sqrt(2)
PLUS_I = np.array([1, 1j], dtype=complex) / np.sqrt(2)


def bloch(state: np.ndarray) -> np.ndarray:
    """Bloch vector (r_x, r_y, r_z) of a one-qubit ket.

    Pure states lie on the unit sphere S^2. Multi-qubit states are rejected.
    """
    ket = np.asarray(state, dtype=complex).reshape(-1)
    if ket.size != 2:
        raise ValueError("state must be a one-qubit ket")
    norm = np.linalg.norm(ket)
    if norm == 0:
        raise ValueError("state has zero norm")
    ket = ket / norm
    rx = 2.0 * float((np.conj(ket[0]) * ket[1]).real)
    ry = 2.0 * float((np.conj(ket[0]) * ket[1]).imag)
    rz = float((np.abs(ket[0]) ** 2 - np.abs(ket[1]) ** 2).real)
    return np.array([rx, ry, rz], dtype=float)


def basis(index: int, qubits: int) -> np.ndarray:
    """Computational-basis ket |index⟩ on `qubits` qubits."""
    if qubits < 1 or not 0 <= index < (1 << qubits):
        raise ValueError("index out of range for the given number of qubits")
    ket = np.zeros(1 << qubits, dtype=complex)
    ket[index] = 1
    return ket


def density(state: np.ndarray) -> np.ndarray:
    """Density operator ρ = |ψ⟩⟨ψ| from a ket |ψ⟩."""
    ket = np.asarray(state, dtype=complex).reshape(-1)
    norm = np.linalg.norm(ket)
    if norm == 0:
        raise ValueError("state has zero norm")
    ket = ket / norm
    return np.outer(ket, np.conj(ket))


def purity(rho: np.ndarray) -> float:
    """Purity γ = Tr(ρ²) of a density operator ρ."""
    mat = np.asarray(rho, dtype=complex)
    if mat.ndim == 1:
        mat = density(mat)
    elif mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("rho must be a square matrix or a ket")
    val = np.trace(mat @ mat)
    return float(val.real)


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
