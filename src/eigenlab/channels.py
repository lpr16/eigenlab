"""One-qubit quantum channels as Kraus maps."""

from __future__ import annotations

from typing import Callable, Sequence

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


def dephasing(rho: np.ndarray, gamma: float, time: float) -> np.ndarray:
    """Closed-form solution of dρ/dt = γ (ZρZ − ρ): ρ₀₁(t) = ρ₀₁(0) e^{−2γt}, diagonal fixed."""
    if gamma < 0:
        raise ValueError("gamma must be nonnegative")
    out = _one_qubit_rho(rho).copy()
    decay = np.exp(-2.0 * float(gamma) * float(time))
    out[0, 1] *= decay
    out[1, 0] *= decay
    return out


def amplitude_damping_evolve(rho: np.ndarray, gamma: float, time: float) -> np.ndarray:
    """Closed-form solution of dρ/dt = γ (σ₋ρσ₊ − ½ {σ₊σ₋, ρ}) with σ₋ = |0⟩⟨1|.

    ρ₁₁(t) = ρ₁₁(0) e^{−γt}, ρ₀₁(t) = ρ₀₁(0) e^{−γt/2}, and ρ₀₀ takes the lost population.
    """
    if gamma < 0:
        raise ValueError("gamma must be nonnegative")
    mat = _one_qubit_rho(rho)
    decay = np.exp(-float(gamma) * float(time))
    out = np.empty_like(mat)
    out[1, 1] = mat[1, 1] * decay
    out[0, 0] = mat[0, 0] + mat[1, 1] * (1.0 - decay)
    out[0, 1] = mat[0, 1] * np.sqrt(decay)
    out[1, 0] = mat[1, 0] * np.sqrt(decay)
    return out


def choi(
    kraus_ops: Sequence[np.ndarray] | Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Choi matrix Σ_ij |i⟩⟨j| ⊗ E(|i⟩⟨j|) of a one-qubit map.

    `kraus_ops` is a list of Kraus operators, or a callable ρ ↦ E(ρ) for a linear map
    without a Kraus form, such as the transpose.
    """
    if callable(kraus_ops):
        channel = kraus_ops
    else:
        ops = list(kraus_ops)

        def channel(rho: np.ndarray) -> np.ndarray:
            return apply_kraus(rho, ops)

    out = np.zeros((4, 4), dtype=complex)
    for i in range(2):
        for j in range(2):
            unit = np.zeros((2, 2), dtype=complex)
            unit[i, j] = 1.0
            out += np.kron(unit, np.asarray(channel(unit), dtype=complex))
    return out
