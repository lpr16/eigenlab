"""Analytic one- and two-electron integrals over s-type Gaussian primitives and STO-3G contractions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import NamedTuple

import numpy as np

# Hehre, Stewart, and Pople, J. Chem. Phys. 51, 2657 (1969), Table I (STO-3G 1s, zeta = 1.0)
STO3G_1S_EXPONENTS: tuple[float, float, float] = (0.109818, 0.405771, 2.22766)
STO3G_1S_COEFFICIENTS: tuple[float, float, float] = (0.444635, 0.535328, 0.154329)


@dataclass(frozen=True)
class ContractedGaussian:
    """Contracted s-type Gaussian basis function."""

    exponents: tuple[float, ...]
    coefficients: tuple[float, ...]
    center: np.ndarray
    zeta: float = 1.0


def _as_center(center: Sequence[float] | np.ndarray | float) -> np.ndarray:
    """Convert a center coordinate (3D vector or scalar along z) to a 3D float array."""
    arr = np.asarray(center, dtype=float)
    if arr.ndim == 0:
        return np.array([0.0, 0.0, float(arr)], dtype=float)
    if arr.shape == (3,):
        return arr
    raise ValueError(f"Center must be a 3-element coordinate or scalar, got shape {arr.shape}")


def gaussian_overlap(
    a: float,
    A: Sequence[float] | np.ndarray | float,
    b: float,
    B: Sequence[float] | np.ndarray | float,
) -> float:
    """Overlap integral S between two normalized s-type Gaussian primitives.

    Each primitive is (2 alpha / pi)^(3/4) exp(-alpha |r - R|^2).
    With p = a + b and mu = a * b / p, the Gaussian product theorem gives:
        S = (4 a b / p^2)^(3/4) exp(-mu |A - B|^2)
    """
    if a <= 0.0 or b <= 0.0:
        raise ValueError("Gaussian exponents must be positive.")
    vec_a = _as_center(A)
    vec_b = _as_center(B)
    diff = vec_a - vec_b
    r2 = float(np.dot(diff, diff))
    p = float(a + b)
    mu = float(a * b) / p
    return ((4.0 * a * b) / (p * p)) ** 0.75 * math.exp(-mu * r2)


def sto3g_1s(
    zeta: float,
    center: Sequence[float] | np.ndarray | float = (0.0, 0.0, 0.0),
) -> ContractedGaussian:
    """Construct an STO-3G 1s contracted Gaussian with Slater exponent ``zeta`` at ``center``.

    Exponents scale as ``zeta**2``; contraction coefficients are independent of ``zeta``.
    Reference: Hehre, Stewart, and Pople, J. Chem. Phys. 51, 2657 (1969).
    """
    if zeta <= 0.0:
        raise ValueError("Slater exponent zeta must be positive.")
    z2 = float(zeta) ** 2
    exponents = tuple(alpha * z2 for alpha in STO3G_1S_EXPONENTS)
    return ContractedGaussian(
        exponents=exponents,
        coefficients=STO3G_1S_COEFFICIENTS,
        center=_as_center(center),
        zeta=float(zeta),
    )


def contracted_overlap(chi_a: ContractedGaussian, chi_b: ContractedGaussian) -> float:
    """Overlap integral between two contracted s-type Gaussians."""
    total = 0.0
    for ca, aa in zip(chi_a.coefficients, chi_a.exponents):
        for cb, ab in zip(chi_b.coefficients, chi_b.exponents):
            total += ca * cb * gaussian_overlap(aa, chi_a.center, ab, chi_b.center)
    return total


def gaussian_kinetic(
    a: float,
    A: Sequence[float] | np.ndarray | float,
    b: float,
    B: Sequence[float] | np.ndarray | float,
) -> float:
    """Kinetic-energy integral T = <a| -1/2 nabla^2 |b> between two normalized s primitives.

    With p = a + b, mu = a * b / p, and overlap S:
        T = mu (3 - 2 mu |A - B|^2) S
    """
    if a <= 0.0 or b <= 0.0:
        raise ValueError("Gaussian exponents must be positive.")
    vec_a = _as_center(A)
    vec_b = _as_center(B)
    diff = vec_a - vec_b
    r2 = float(np.dot(diff, diff))
    p = float(a + b)
    mu = float(a * b) / p
    s = gaussian_overlap(a, vec_a, b, vec_b)
    return mu * (3.0 - 2.0 * mu * r2) * s


def contracted_kinetic(chi_a: ContractedGaussian, chi_b: ContractedGaussian) -> float:
    """Kinetic-energy integral between two contracted s-type Gaussians."""
    total = 0.0
    for ca, aa in zip(chi_a.coefficients, chi_a.exponents):
        for cb, ab in zip(chi_b.coefficients, chi_b.exponents):
            total += ca * cb * gaussian_kinetic(aa, chi_a.center, ab, chi_b.center)
    return total


def boys0(t: float) -> float:
    """Zeroth-order Boys function F_0(t) = int_0^1 exp(-t u^2) du.

    For t > 0, F_0(t) = 1/2 sqrt(pi / t) erf(sqrt(t)), and F_0(0) = 1.
    """
    if t < 0.0:
        raise ValueError("Boys function argument t must be nonnegative.")
    if t < 1e-12:
        return 1.0 - t / 3.0 + (t * t) / 10.0
    sqrt_t = math.sqrt(t)
    return 0.5 * math.sqrt(math.pi / t) * math.erf(sqrt_t)


def gaussian_nuclear_attraction(
    a: float,
    A: Sequence[float] | np.ndarray | float,
    b: float,
    B: Sequence[float] | np.ndarray | float,
    Z: float,
    C: Sequence[float] | np.ndarray | float,
) -> float:
    """Nuclear-attraction integral V = <a| -Z / |r - C| |b> between two normalized s primitives.

    With p = a + b, mu = a * b / p, P = (a A + b B) / p, and K = exp(-mu |A - B|^2):
        V = -Z (2 pi / p) K F_0(p |P - C|^2) (4 a b / pi^2)^(3/4)
    """
    if a <= 0.0 or b <= 0.0:
        raise ValueError("Gaussian exponents must be positive.")
    vec_a = _as_center(A)
    vec_b = _as_center(B)
    vec_c = _as_center(C)
    diff_ab = vec_a - vec_b
    r2_ab = float(np.dot(diff_ab, diff_ab))
    p = float(a + b)
    mu = float(a * b) / p
    vec_p = (a * vec_a + b * vec_b) / p
    diff_pc = vec_p - vec_c
    r2_pc = float(np.dot(diff_pc, diff_pc))
    k_ab = math.exp(-mu * r2_ab)
    norm_ab = ((4.0 * a * b) / (math.pi * math.pi)) ** 0.75
    return -float(Z) * (2.0 * math.pi / p) * k_ab * boys0(p * r2_pc) * norm_ab


def contracted_nuclear_attraction(
    chi_a: ContractedGaussian,
    chi_b: ContractedGaussian,
    Z: float,
    C: Sequence[float] | np.ndarray | float,
) -> float:
    """Nuclear-attraction integral of charge Z at C between two contracted s-type Gaussians."""
    vec_c = _as_center(C)
    total = 0.0
    for ca, aa in zip(chi_a.coefficients, chi_a.exponents):
        for cb, ab in zip(chi_b.coefficients, chi_b.exponents):
            total += ca * cb * gaussian_nuclear_attraction(aa, chi_a.center, ab, chi_b.center, Z, vec_c)
    return total


def gaussian_eri(
    a: float,
    A: Sequence[float] | np.ndarray | float,
    b: float,
    B: Sequence[float] | np.ndarray | float,
    c: float,
    C: Sequence[float] | np.ndarray | float,
    d: float,
    D: Sequence[float] | np.ndarray | float,
) -> float:
    """Two-electron repulsion integral (ab|cd) in chemist notation for four normalized s primitives.

    With p = a + b, q = c + d, P = (a A + b B)/p, Q = (c C + d D)/q,
    K_ab = exp(-(ab/p)|A - B|^2), and K_cd = exp(-(cd/q)|C - D|^2):
        (ab|cd) = [2 pi^(5/2) / (p q sqrt(p + q))] K_ab K_cd F_0(pq/(p+q) |P - Q|^2)
                  * (16 a b c d / pi^4)^(3/4)
    """
    if a <= 0.0 or b <= 0.0 or c <= 0.0 or d <= 0.0:
        raise ValueError("Gaussian exponents must be positive.")
    vec_a = _as_center(A)
    vec_b = _as_center(B)
    vec_c = _as_center(C)
    vec_d = _as_center(D)

    p = float(a + b)
    q = float(c + d)
    mu_ab = float(a * b) / p
    mu_cd = float(c * d) / q

    diff_ab = vec_a - vec_b
    diff_cd = vec_c - vec_d
    r2_ab = float(np.dot(diff_ab, diff_ab))
    r2_cd = float(np.dot(diff_cd, diff_cd))

    vec_p = (a * vec_a + b * vec_b) / p
    vec_q = (c * vec_c + d * vec_d) / q
    diff_pq = vec_p - vec_q
    r2_pq = float(np.dot(diff_pq, diff_pq))

    k_ab = math.exp(-mu_ab * r2_ab)
    k_cd = math.exp(-mu_cd * r2_cd)
    norm_abcd = ((16.0 * a * b * c * d) / (math.pi**4)) ** 0.75
    prefactor = 2.0 * (math.pi**2.5) / (p * q * math.sqrt(p + q))
    return prefactor * k_ab * k_cd * boys0((p * q / (p + q)) * r2_pq) * norm_abcd


gaussian_electron_repulsion = gaussian_eri


def contracted_eri(
    chi_a: ContractedGaussian,
    chi_b: ContractedGaussian,
    chi_c: ContractedGaussian,
    chi_d: ContractedGaussian,
) -> float:
    """Two-electron repulsion integral (ab|cd) in chemist notation for four contracted s Gaussians."""
    total = 0.0
    for ca, aa in zip(chi_a.coefficients, chi_a.exponents):
        for cb, ab in zip(chi_b.coefficients, chi_b.exponents):
            for cc, ac in zip(chi_c.coefficients, chi_c.exponents):
                for cd, ad in zip(chi_d.coefficients, chi_d.exponents):
                    total += (
                        ca
                        * cb
                        * cc
                        * cd
                        * gaussian_eri(
                            aa, chi_a.center, ab, chi_b.center, ac, chi_c.center, ad, chi_d.center
                        )
                    )
    return total


contracted_electron_repulsion = contracted_eri


class AOIntegrals(NamedTuple):
    """Atomic-orbital integral bundle ``(S, h, eri, v_nuc)``."""

    S: np.ndarray
    h: np.ndarray
    eri: np.ndarray
    v_nuc: float

    @property
    def overlap(self) -> np.ndarray:
        return self.S

    @property
    def core(self) -> np.ndarray:
        return self.h

    @property
    def nuclear_repulsion(self) -> float:
        return self.v_nuc


def ao_integrals(
    atoms: Sequence[tuple[float, float, Sequence[float] | np.ndarray | float]],
) -> AOIntegrals:
    """Compute STO-3G 1s atomic-orbital integrals ``(S, h, eri, V_nuc)`` for ``atoms``.

    Each entry in ``atoms`` is ``(Z, zeta, position)``.
    Returns ``AOIntegrals(S, h, eri, v_nuc)``, which unpacks as a 4-tuple
    ``(S, h, eri, V_nuc)`` with ``h = T + V`` and ``V_nuc = sum_{i < j} Z_i Z_j / |R_i - R_j|``.
    """
    n = len(atoms)
    if n == 0:
        raise ValueError("atoms must be non-empty.")

    charges: list[float] = []
    centers: list[np.ndarray] = []
    basis_fns: list[ContractedGaussian] = []

    for z_val, zeta, pos in atoms:
        c_vec = _as_center(pos)
        charges.append(float(z_val))
        centers.append(c_vec)
        basis_fns.append(sto3g_1s(float(zeta), c_vec))

    v_nuc = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.linalg.norm(centers[i] - centers[j]))
            if dist <= 1e-12:
                raise ValueError(f"Atoms {i} and {j} coincide at {centers[i]}.")
            v_nuc += charges[i] * charges[j] / dist

    s_mat = np.zeros((n, n), dtype=float)
    h_mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            s_ij = contracted_overlap(basis_fns[i], basis_fns[j])
            t_ij = contracted_kinetic(basis_fns[i], basis_fns[j])
            v_ij = sum(
                contracted_nuclear_attraction(basis_fns[i], basis_fns[j], charges[k], centers[k])
                for k in range(n)
            )
            h_ij = t_ij + v_ij
            s_mat[i, j] = s_mat[j, i] = s_ij
            h_mat[i, j] = h_mat[j, i] = h_ij

    eri = np.zeros((n, n, n, n), dtype=float)
    computed: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}
    for p in range(n):
        for q in range(p, n):
            pair_pq = (p, q)
            for r in range(n):
                for s in range(r, n):
                    pair_rs = (r, s)
                    canon = (pair_pq, pair_rs) if pair_pq <= pair_rs else (pair_rs, pair_pq)
                    if canon not in computed:
                        computed[canon] = contracted_eri(
                            basis_fns[p], basis_fns[q], basis_fns[r], basis_fns[s]
                        )
                    val = computed[canon]
                    for idx in (
                        (p, q, r, s),
                        (q, p, r, s),
                        (p, q, s, r),
                        (q, p, s, r),
                        (r, s, p, q),
                        (s, r, p, q),
                        (r, s, q, p),
                        (s, r, q, p),
                    ):
                        eri[idx] = val

    return AOIntegrals(S=s_mat, h=h_mat, eri=eri, v_nuc=v_nuc)


def h2_symmetry_orbitals(S: np.ndarray | float) -> np.ndarray:
    """Return the 2x2 minimal-basis symmetry-adapted MO coefficient matrix C = [sigma_g, sigma_u]."""
    arr = np.asarray(S, dtype=float)
    if arr.ndim == 0:
        s00 = 1.0
        s01 = float(arr)
    elif arr.shape == (2, 2):
        s00 = float(arr[0, 0])
        s01 = float(arr[0, 1])
    else:
        raise ValueError(f"Expected scalar overlap or 2x2 overlap matrix, got shape {arr.shape}")
    cg = 1.0 / math.sqrt(2.0 * (s00 + s01))
    cu = 1.0 / math.sqrt(2.0 * (s00 - s01))
    return np.array([[cg, cu], [cg, -cu]], dtype=float)


def mo_transform(
    C: np.ndarray,
    h: np.ndarray,
    eri: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform AO one- and two-electron integrals ``(h, eri)`` into the MO basis ``C`` (chemist notation)."""
    c_mat = np.asarray(C, dtype=float)
    h_ao = np.asarray(h, dtype=float)
    eri_ao = np.asarray(eri, dtype=float)
    h_mo = c_mat.T @ h_ao @ c_mat
    eri_mo = np.einsum("mp,nq,lr,ks,mnlk->pqrs", c_mat, c_mat, c_mat, c_mat, eri_ao)
    return h_mo, eri_mo


def lowdin(S: np.ndarray) -> np.ndarray:
    """Return the symmetric orthogonalization matrix ``X = S^(-1/2)`` via ``eigh``.

    Raises ``ValueError`` if ``S`` has an eigenvalue below ``1e-10``.
    """
    s_mat = np.asarray(S, dtype=float)
    if s_mat.ndim != 2 or s_mat.shape[0] != s_mat.shape[1]:
        raise ValueError(f"Overlap matrix S must be square, got shape {s_mat.shape}")
    if not np.allclose(s_mat, s_mat.T, atol=1e-10):
        raise ValueError("Overlap matrix S must be symmetric.")
    evals, evecs = np.linalg.eigh(s_mat)
    if np.any(evals < 1e-10):
        raise ValueError(f"Overlap matrix S has an eigenvalue below 1e-10: {evals.min()}")
    return (evecs * (1.0 / np.sqrt(evals))) @ evecs.T







