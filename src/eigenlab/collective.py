"""Collective spin operators, spin matrices, and Dicke states on the symmetric subspace."""

from __future__ import annotations

import math
import numpy as np

from eigenlab.hamiltonian import Hamiltonian


def collective_spin(n: int) -> tuple[Hamiltonian, Hamiltonian, Hamiltonian]:
    """Return collective spin operators J_x, J_y, J_z = ∑_{i=0}^{n-1} σ_i / 2 as Hamiltonians/term lists."""
    if n < 1:
        raise ValueError("n must be at least 1")
    jx_terms: list[tuple[complex, str]] = []
    jy_terms: list[tuple[complex, str]] = []
    jz_terms: list[tuple[complex, str]] = []
    for i in range(n):
        lbl_x = ["I"] * n
        lbl_x[i] = "X"
        lbl_y = ["I"] * n
        lbl_y[i] = "Y"
        lbl_z = ["I"] * n
        lbl_z[i] = "Z"
        jx_terms.append((0.5, "".join(lbl_x)))
        jy_terms.append((0.5, "".join(lbl_y)))
        jz_terms.append((0.5, "".join(lbl_z)))
    return Hamiltonian(jx_terms), Hamiltonian(jy_terms), Hamiltonian(jz_terms)


def spin_matrices(j: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (J_x, J_y, J_z) matrices of dimension 2j + 1 in basis m = j, j-1, ..., -j.

    J_z = diag(j, ..., -j), and J_+ has elements √(j(j+1) - m(m+1)).
    """
    if j < 0 or not np.isclose(round(2 * j), 2 * j):
        raise ValueError("j must be a non-negative half-integer or integer")
    d = int(round(2 * j + 1))
    m_vals = np.array([j - k for k in range(d)], dtype=float)
    j_z = np.diag(m_vals).astype(complex)

    j_plus = np.zeros((d, d), dtype=complex)
    for k in range(1, d):
        m = j - k
        val = np.sqrt(j * (j + 1) - m * (m + 1))
        j_plus[k - 1, k] = val
    j_minus = j_plus.conj().T
    j_x = (j_plus + j_minus) / 2.0
    j_y = (j_plus - j_minus) / 2.0j
    return j_x, j_y, j_z


def dicke(n: int, k: int) -> np.ndarray:
    """Return the normalized Dicke state |n, k⟩ = (1/√(n choose k)) ∑_{weight(x)=k} |x⟩."""
    if not (0 <= k <= n):
        raise ValueError(f"k must be between 0 and n, got k={k}, n={n}")
    dim = 1 << n
    vec = np.zeros(dim, dtype=complex)
    norm = 1.0 / np.sqrt(math.comb(n, k))
    for x in range(dim):
        if bin(x).count("1") == k:
            vec[x] = norm
    return vec
