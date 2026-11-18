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


def one_axis_state(n: int, chi: float, time: float) -> np.ndarray:
    """Evolve |+⟩^{⊗n} under H = χ J_z² inside the symmetric subspace.

    The initial state is the spin coherent state along x, written in the Dicke
    basis k = 0…n (m = n/2 − k). J_z² is diagonal there, with entries m².
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    amps = np.empty(n + 1, dtype=complex)
    amps[0] = 2.0 ** (-n / 2.0)
    for k in range(n):
        amps[k + 1] = amps[k] * math.sqrt((n - k) / (k + 1.0))
    j = n / 2.0
    ms = j - np.arange(n + 1, dtype=float)
    phases = np.exp(-1j * float(chi) * (ms ** 2) * float(time))
    return amps * phases


def one_axis_squeezing(
    n: int, chi: float, time: float
) -> tuple[float, float, float]:
    """Wineland squeezing of one-axis twisting, and the Pezzé–Smerzi bound.

    Returns ``(ξ², F, n/ξ²)``. ``V_min`` is the smaller eigenvalue of the
    symmetrized covariance of ``J_y`` and ``J_z``, ``s`` its direction, and
    ``ξ² = n V_min / ⟨J_x⟩²``. ``F = 4 Var(J_⊥)`` with ``J_⊥`` along ``x̂ × s``.
    """
    psi = one_axis_state(n, chi, time)
    jx, jy, jz = spin_matrices(n / 2.0)

    def expect(op: np.ndarray) -> complex:
        return np.vdot(psi, op @ psi)

    mean_x = float(np.real(expect(jx)))
    mean_y = float(np.real(expect(jy)))
    mean_z = float(np.real(expect(jz)))
    ops = (jy, jz)
    means = (mean_y, mean_z)
    cov = np.zeros((2, 2), dtype=float)
    for a in range(2):
        for b in range(a, 2):
            anti = 0.5 * (expect(ops[a] @ ops[b]) + expect(ops[b] @ ops[a]))
            val = float(np.real(anti)) - means[a] * means[b]
            cov[a, b] = cov[b, a] = val
    evals, evecs = np.linalg.eigh(cov)
    v_min = float(evals[0])
    sy, sz = (float(evecs[0, 0]), float(evecs[1, 0]))
    # x̂ × (sy ŷ + sz ẑ) = sy ẑ − sz ŷ
    j_perp = -sz * jy + sy * jz
    mean_p = float(np.real(expect(j_perp)))
    var_p = float(np.real(expect(j_perp @ j_perp))) - mean_p * mean_p
    fisher = 4.0 * var_p
    xi2 = n * v_min / (mean_x * mean_x)
    return xi2, fisher, n / xi2


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
