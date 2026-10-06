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
