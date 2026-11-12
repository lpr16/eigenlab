"""The Clifford group, randomized benchmarking, and magic."""

from __future__ import annotations

from typing import Callable

import numpy as np


I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0)
S = np.array([[1, 0], [0, 1j]], dtype=complex)


def canonicalize_phase(u: np.ndarray, tol: float = 1e-10) -> np.ndarray:

    """Fix global phase of matrix u so that its first nonzero entry is real and positive."""
    mat = np.asarray(u, dtype=complex)
    for val in mat.ravel():
        if abs(val) > tol:
            phase = val / abs(val)
            return mat * np.conj(phase)
    return mat


def one_qubit_cliffords() -> list[np.ndarray]:
    """Return the 24 one-qubit Clifford group unitaries modulo global phase.

    Each unitary has its phase fixed so that the first nonzero entry is real and positive.
    """
    h_gate = canonicalize_phase(H)
    s_gate = canonicalize_phase(S)

    group: list[np.ndarray] = [canonicalize_phase(I)]
    gens = [h_gate, s_gate]

    idx = 0
    while idx < len(group):
        curr = group[idx]
        idx += 1
        for g in gens:
            prod = canonicalize_phase(curr @ g)
            if not any(np.allclose(prod, elem, atol=1e-8) for elem in group):
                group.append(prod)

    return group


def twirl(channel: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    """Twirl a one-qubit channel over the 24-element Clifford group.

    Returns the twirled channel ρ ↦ (1/24) ∑_{C ∈ G} C† channel(C ρ C†) C.
    """
    cliffords = one_qubit_cliffords()

    def twirled_channel(rho: np.ndarray) -> np.ndarray:
        mat = np.asarray(rho, dtype=complex)
        if mat.ndim == 1:
            from eigenlab.states import density

            mat = density(mat)
        out = np.zeros_like(mat)
        for c in cliffords:
            out += c.conj().T @ channel(c @ mat @ c.conj().T) @ c
        return out / len(cliffords)

    return twirled_channel

