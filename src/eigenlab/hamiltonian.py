"""Spin Hamiltonians and their exact spectra."""

from __future__ import annotations

import numpy as np

from eigenlab.pauli import pauli


def heisenberg_dimer() -> np.ndarray:
    """H = X⊗X + Y⊗Y + Z⊗Z. Singlet at −3, triplet at +1."""
    return pauli("XX") + pauli("YY") + pauli("ZZ")


def transverse_ising(qubits: int, coupling: float = 1.0, field: float = 1.0) -> np.ndarray:
    """Open-chain transverse-field Ising, H = −J Σ Z_i Z_{i+1} − h Σ X_i."""
    if qubits < 2:
        raise ValueError("qubits must be at least 2")
    dim = 1 << qubits
    hamiltonian = np.zeros((dim, dim), dtype=complex)
    for site in range(qubits - 1):
        label = ["I"] * qubits
        label[site] = "Z"
        label[site + 1] = "Z"
        hamiltonian -= coupling * pauli("".join(label))
    for site in range(qubits):
        label = ["I"] * qubits
        label[site] = "X"
        hamiltonian -= field * pauli("".join(label))
    return hamiltonian


def spectrum(hamiltonian: np.ndarray) -> np.ndarray:
    """Ascending eigenvalues of a Hermitian operator."""
    values = np.linalg.eigvalsh(np.asarray(hamiltonian, dtype=complex))
    return np.real(values)
