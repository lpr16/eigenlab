"""One-qubit quantum channels as Kraus maps."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from eigenlab.states import density


def _one_qubit_rho(rho: np.ndarray) -> np.ndarray:
    mat = np.asarray(rho, dtype=complex)
    if mat.ndim == 1:
        mat = density(mat)
    if mat.shape != (2, 2):
        raise ValueError("rho must be a one-qubit density matrix or ket")
    return mat


def apply_kraus(rho: np.ndarray, kraus_ops: Sequence[np.ndarray]) -> np.ndarray:
    """E(ρ) = Σ_k E_k ρ E_k†."""
    mat = np.asarray(rho, dtype=complex)
    if mat.ndim == 1:
        mat = density(mat)
    out = np.zeros_like(mat)
    for op in kraus_ops:
        k = np.asarray(op, dtype=complex)
        out += k @ mat @ k.conj().T
    return out


def amplitude_damping_kraus(p: float) -> list[np.ndarray]:
    """Kraus operators E0 = [[1, 0], [0, √(1−p)]] and E1 = [[0, √p], [0, 0]]."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    e0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - p)]], dtype=complex)
    e1 = np.array([[0.0, np.sqrt(p)], [0.0, 0.0]], dtype=complex)
    return [e0, e1]


def amplitude_damping(rho: np.ndarray, p: float) -> np.ndarray:
    """One-qubit amplitude damping: |1⟩ decays to |0⟩ with probability p."""
    return apply_kraus(_one_qubit_rho(rho), amplitude_damping_kraus(p))


def depolarize(rho: np.ndarray, p: float) -> np.ndarray:
    """One-qubit depolarizing channel (1−p) ρ + (p/3) (XρX + YρY + ZρZ), 0 ≤ p ≤ 1."""
    from eigenlab.pauli import X, Y, Z

    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    mat = _one_qubit_rho(rho)
    return (1.0 - p) * mat + (p / 3.0) * (X @ mat @ X + Y @ mat @ Y + Z @ mat @ Z)
