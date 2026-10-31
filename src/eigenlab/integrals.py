"""Analytic one- and two-electron integrals over s-type Gaussian primitives and STO-3G contractions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

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


