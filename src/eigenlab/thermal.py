"""Thermal Gibbs states, partition functions, and thermodynamic entropies."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from eigenlab.fermions import kitaev_bdg_matrix
from eigenlab.hamiltonian import Hamiltonian, eigensystem, transverse_ising_bdg_matrix


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


def bdg_partition_function(bdg_matrix: np.ndarray, beta: float) -> float:
    """Return ``Pi_k 2 cosh(beta eps_k / 2)`` from the n nonnegative eigenvalues of a 2n x 2n BdG matrix."""
    b_val = float(beta)
    if b_val < 0.0:
        raise ValueError(f"Inverse temperature beta must be nonnegative, got {beta}")
    bdg = np.asarray(bdg_matrix, dtype=float)
    if bdg.ndim != 2 or bdg.shape[0] != bdg.shape[1] or bdg.shape[0] % 2 != 0:
        raise ValueError(f"Expected 2n x 2n BdG matrix, got shape {bdg.shape}")
    n = bdg.shape[0] // 2
    eps = np.linalg.eigvalsh(bdg)[n:]
    return float(np.prod(2.0 * np.cosh(0.5 * b_val * eps)))


def ising_partition_function(
    n: int,
    beta: float,
    coupling: float = 1.0,
    field: float = 1.0,
) -> float:
    """Free-fermion partition function of the open transverse Ising chain from its BdG modes."""
    return bdg_partition_function(
        transverse_ising_bdg_matrix(n, coupling=coupling, field=field),
        beta,
    )


def kitaev_partition_function(
    n: int,
    beta: float,
    t: float = 1.0,
    delta: float = 1.0,
    mu: float = 0.0,
) -> float:
    """Free-fermion partition function of the open Kitaev chain from its BdG modes."""
    return bdg_partition_function(
        kitaev_bdg_matrix(n, t=t, delta=delta, mu=mu),
        beta,
    )


__all__ = [
    "bdg_partition_function",
    "gibbs",
    "ising_partition_function",
    "kitaev_partition_function",
    "partition_function",
    "trace_distance",
]
