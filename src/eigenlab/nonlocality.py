"""Bell operators whose quantum value beats every local assignment."""

from __future__ import annotations

import itertools

import numpy as np

from eigenlab.hamiltonian import Hamiltonian
from eigenlab.pauli import X, Y, Z


def mermin_operator(n: int) -> Hamiltonian:
    """Real part of ∏_k (X_k + i Y_k).

    The expansion keeps every X/Y string with an even number of Ys, and the
    sign of that string is (−1)^{#Y / 2}.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    terms: list[tuple[float, str]] = []
    for letters in itertools.product("XY", repeat=n):
        n_y = letters.count("Y")
        if n_y % 2:
            continue
        sign = float((-1) ** (n_y // 2))
        terms.append((sign, "".join(letters)))
    return Hamiltonian(terms).simplify()


# Rows XI IX XX; IZ ZI ZZ; XZ ZX YY.
PERES_MERMIN_SQUARE: tuple[tuple[str, str, str], ...] = (
    ("XI", "IX", "XX"),
    ("IZ", "ZI", "ZZ"),
    ("XZ", "ZX", "YY"),
)


def correlation_matrix(rho: np.ndarray) -> np.ndarray:
    """T_ij = Tr(ρ σ_i ⊗ σ_j) for σ in (X, Y, Z)."""
    state = np.asarray(rho, dtype=complex)
    if state.shape != (4, 4):
        raise ValueError(f"correlation_matrix expects a two-qubit state, got {state.shape}")
    paulis = (X, Y, Z)
    corr = np.zeros((3, 3), dtype=float)
    for i, left in enumerate(paulis):
        for j, right in enumerate(paulis):
            op = np.kron(left, right)
            corr[i, j] = float(np.real(np.trace(state @ op)))
    return corr


def chsh_horodecki(rho: np.ndarray) -> float:
    """Largest CHSH value 2√M, with M the sum of the two largest eigenvalues of TᵀT.

    Horodecki, Horodecki and Horodecki, Phys. Lett. A 200, 340 (1995).
    """
    corr = correlation_matrix(rho)
    evals = np.linalg.eigvalsh(corr.T @ corr)
    largest = float(evals[-1] + evals[-2])
    return 2.0 * float(np.sqrt(max(0.0, largest)))


def mermin_three() -> Hamiltonian:
    """Three-qubit Mermin operator ``XXX − XYY − YXY − YYX``."""
    return Hamiltonian(
        [(1.0, "XXX"), (-1.0, "XYY"), (-1.0, "YXY"), (-1.0, "YYX")]
    ).simplify()
