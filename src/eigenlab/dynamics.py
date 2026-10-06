"""Time evolution and Trotter approximations."""

from __future__ import annotations

import numpy as np

from eigenlab.hamiltonian import Hamiltonian, eigensystem


def evolve(
    H: np.ndarray | Hamiltonian | float | int,
    state: np.ndarray,
    time: float,
) -> np.ndarray:
    """Exact unitary time evolution |ψ(t)⟩ = exp(-i H t) |ψ(0)⟩ via spectral decomposition.

    U = V exp(-i E t) V†
    """
    ket = np.asarray(state, dtype=complex).reshape(-1)
    if isinstance(H, (int, float)):
        if H == 0:
            return ket.copy()
        # Non-zero scalar multiple of identity
        return np.exp(-1j * float(H) * float(time)) * ket

    energies, V = eigensystem(H)
    phases = np.exp(-1j * energies * float(time))
    c = V.conj().T @ ket
    return V @ (phases * c)


def trotter_step(
    H: Hamiltonian | Sequence[tuple[complex | float, str]],
    state: np.ndarray,
    dt: float,
) -> np.ndarray:
    """One first-order Trotter step |ψ'⟩ = ∏ exp(-i c_k P_k Δt) |ψ⟩.

    Uses P_k² = I, so each factor is cos(θ) I - i sin(θ) P_k with θ = c_k Δt.
    """
    from eigenlab.pauli import pauli

    terms = H.terms if isinstance(H, Hamiltonian) else list(H)
    ket = np.asarray(state, dtype=complex).reshape(-1).copy()
    for coeff, label in terms:
        theta = complex(coeff) * float(dt)
        p_mat = pauli(label)
        ket = np.cos(theta) * ket - 1j * np.sin(theta) * (p_mat @ ket)
    return ket


def trotter_unitary(
    H: Hamiltonian | Sequence[tuple[complex | float, str]],
    dt: float,
) -> np.ndarray:
    """One first-order Trotter unitary U = ∏ exp(-i c_k P_k Δt)."""
    from eigenlab.pauli import pauli

    terms = H.terms if isinstance(H, Hamiltonian) else list(H)
    if not terms:
        raise ValueError("term list cannot be empty")
    qubits = len(terms[0][1])
    dim = 1 << qubits
    U = np.eye(dim, dtype=complex)
    for coeff, label in terms:
        theta = complex(coeff) * float(dt)
        p_mat = pauli(label)
        factor = np.cos(theta) * np.eye(dim, dtype=complex) - 1j * np.sin(theta) * p_mat
        U = factor @ U
    return U


def strang_step(
    H: Hamiltonian | Sequence[tuple[complex | float, str]],
    state: np.ndarray,
    dt: float,
) -> np.ndarray:
    """One second-order Strang splitting (symmetric Trotter) step on a state ket."""
    from eigenlab.pauli import pauli

    terms = H.terms if isinstance(H, Hamiltonian) else list(H)
    ket = np.asarray(state, dtype=complex).reshape(-1).copy()
    if len(terms) <= 1:
        return trotter_step(terms, ket, dt)
    for coeff, label in terms[:-1]:
        theta = complex(coeff) * (float(dt) / 2.0)
        p_mat = pauli(label)
        ket = np.cos(theta) * ket - 1j * np.sin(theta) * (p_mat @ ket)
    c_last, l_last = terms[-1]
    theta = complex(c_last) * float(dt)
    p_mat = pauli(l_last)
    ket = np.cos(theta) * ket - 1j * np.sin(theta) * (p_mat @ ket)
    for coeff, label in reversed(terms[:-1]):
        theta = complex(coeff) * (float(dt) / 2.0)
        p_mat = pauli(label)
        ket = np.cos(theta) * ket - 1j * np.sin(theta) * (p_mat @ ket)
    return ket


def strang_unitary(
    H: Hamiltonian | Sequence[tuple[complex | float, str]],
    dt: float,
) -> np.ndarray:
    """One second-order Strang splitting (symmetric Trotter) unitary matrix."""
    from eigenlab.pauli import pauli

    terms = H.terms if isinstance(H, Hamiltonian) else list(H)
    if not terms:
        raise ValueError("term list cannot be empty")
    qubits = len(terms[0][1])
    dim = 1 << qubits
    if len(terms) <= 1:
        return trotter_unitary(terms, dt)
    U = np.eye(dim, dtype=complex)
    for coeff, label in terms[:-1]:
        theta = complex(coeff) * (float(dt) / 2.0)
        p_mat = pauli(label)
        factor = np.cos(theta) * np.eye(dim, dtype=complex) - 1j * np.sin(theta) * p_mat
        U = factor @ U
    c_last, l_last = terms[-1]
    theta = complex(c_last) * float(dt)
    p_mat = pauli(l_last)
    factor = np.cos(theta) * np.eye(dim, dtype=complex) - 1j * np.sin(theta) * p_mat
    U = factor @ U
    for coeff, label in reversed(terms[:-1]):
        theta = complex(coeff) * (float(dt) / 2.0)
        p_mat = pauli(label)
        factor = np.cos(theta) * np.eye(dim, dtype=complex) - 1j * np.sin(theta) * p_mat
        U = factor @ U
    return U


def unitary(
    H: np.ndarray | Hamiltonian,
    time: float,
) -> np.ndarray:
    """Exact unitary time-evolution operator U(t) = exp(-i H t)."""
    energies, V = eigensystem(H)
    phases = np.exp(-1j * energies * float(time))
    return V @ np.diag(phases) @ V.conj().T
