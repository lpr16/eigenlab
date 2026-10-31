"""Thermal Gibbs states, partition functions, and thermodynamic entropies."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from eigenlab.hamiltonian import Hamiltonian, eigensystem


def gibbs(
    H: Hamiltonian | Sequence[tuple[complex | float, str]] | np.ndarray,
    beta: float,
) -> np.ndarray:
    """Return the thermal Gibbs density matrix ``exp(-beta H) / Tr exp(-beta H)`` via ``eigh``."""
    b_val = float(beta)
    if b_val < 0.0:
        raise ValueError(f"Inverse temperature beta must be nonnegative, got {beta}")
    evals, evecs = eigensystem(H)
    shifted = evals - evals[0]
    weights = np.exp(-b_val * shifted)
    probs = weights / np.sum(weights)
    return (evecs * probs) @ evecs.conj().T


def partition_function(
    H: Hamiltonian | Sequence[tuple[complex | float, str]] | np.ndarray,
    beta: float,
) -> float:
    """Return the canonical partition function ``Z(beta) = Tr exp(-beta H)`` via ``eigh``."""
    b_val = float(beta)
    if b_val < 0.0:
        raise ValueError(f"Inverse temperature beta must be nonnegative, got {beta}")
    evals, _ = eigensystem(H)
    return float(np.sum(np.exp(-b_val * evals)))


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Return the trace distance ``1/2 ||rho - sigma||_1`` between two Hermitian density matrices."""
    diff = np.asarray(rho, dtype=complex) - np.asarray(sigma, dtype=complex)
    return 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(diff))))


__all__ = [
    "gibbs",
    "partition_function",
    "trace_distance",
]
