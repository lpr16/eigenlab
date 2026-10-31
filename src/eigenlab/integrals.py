"""Analytic one- and two-electron integrals over s-type Gaussian primitives and STO-3G contractions."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


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
