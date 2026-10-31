"""Quantum error-correcting stabilizer codes and noise channels."""

from __future__ import annotations

import numpy as np

from eigenlab.pauli import apply_pauli, pauli_comm, pauli_mult
from eigenlab.states import density


def _as_density_matrix(state: np.ndarray) -> tuple[np.ndarray, int]:
    """Convert a ket or density matrix to a (2^n, 2^n) complex array and return (rho, n)."""
    mat = np.asarray(state, dtype=complex)
    if mat.ndim == 1:
        mat = density(mat)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Expected square density matrix or ket, got shape {mat.shape}")
    dim = mat.shape[0]
    if dim < 2 or (dim & (dim - 1)) != 0:
        raise ValueError(f"Dimension {dim} is not a positive power of 2")
    n = dim.bit_length() - 1
    if n > 8:
        raise ValueError(f"Matrices only supported for n <= 8 qubits, got n={n}")
    return mat.copy(), n


def bit_flip_channel(rho: np.ndarray, p: float) -> np.ndarray:
    """Apply independent single-qubit bit-flip channels with probability ``p`` to ``n <= 8`` qubits.

    Each qubit independently receives Kraus operators ``sqrt(1 - p) I`` and ``sqrt(p) X``.
    """
    p_val = float(p)
    if not 0.0 <= p_val <= 1.0:
        raise ValueError(f"Probability p must be in [0, 1], got {p}")
    out, n = _as_density_matrix(rho)
    dim = 1 << n
    basis_indices = np.arange(dim, dtype=complex)
    for q in range(n):
        label = "I" * q + "X" + "I" * (n - 1 - q)
        perm = np.real(apply_pauli(label, basis_indices)).astype(int)
        out = (1.0 - p_val) * out + p_val * out[np.ix_(perm, perm)]
    return out


__all__ = [
    "apply_pauli",
    "bit_flip_channel",
    "pauli_comm",
    "pauli_mult",
]
