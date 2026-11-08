"""Truncated quantum harmonic oscillator and Jaynes-Cummings model."""

from __future__ import annotations

import numpy as np


def _validate_dim(d: int, max_dim: int = 256) -> int:
    dim = int(d)
    if not 2 <= dim <= max_dim:
        raise ValueError(f"Truncation dimension d must satisfy 2 <= d <= {max_dim}, got d={d}")
    return dim


def fock(n: int, d: int) -> np.ndarray:
    """Return the Fock basis ket ``|n>`` in a ``d``-level truncated oscillator."""
    dim = _validate_dim(d)
    idx = int(n)
    if not 0 <= idx < dim:
        raise ValueError(f"Fock index n={n} out of range for d={dim}")
    ket = np.zeros(dim, dtype=complex)
    ket[idx] = 1.0
    return ket


def annihilation(d: int) -> np.ndarray:
    """Return the ``d x d`` truncated annihilation operator ``a`` with ``a|n> = sqrt(n)|n-1>``."""
    dim = _validate_dim(d)
    mat = np.zeros((dim, dim), dtype=complex)
    n_vals = np.arange(1, dim, dtype=float)
    mat[np.arange(dim - 1), np.arange(1, dim)] = np.sqrt(n_vals)
    return mat


def creation(d: int) -> np.ndarray:
    """Return the ``d x d`` truncated creation operator ``a^dag``."""
    return annihilation(d).conj().T


def number(d: int) -> np.ndarray:
    """Return the ``d x d`` number operator ``a^dag a = diag(0, ..., d-1)``."""
    dim = _validate_dim(d)
    return np.diag(np.arange(dim, dtype=complex))


def coherent(alpha: complex | float, d: int) -> np.ndarray:
    """Return the unnormalized truncated coherent state ``c_n = exp(-|alpha|^2/2) alpha^n / sqrt(n!)``."""
    import math

    dim = _validate_dim(d)
    alp = complex(alpha)
    pref = np.exp(-0.5 * (abs(alp) ** 2))
    return np.array(
        [pref * (alp**n) / math.sqrt(math.factorial(n)) for n in range(dim)],
        dtype=complex,
    )


def jaynes_cummings(
    d: int,
    omega_c: float = 1.0,
    omega_q: float = 1.0,
    g: float = 0.1,
) -> np.ndarray:
    """Return the ``2d x 2d`` Jaynes-Cummings Hamiltonian ``H = omega_c a^dag a ⊗ I - (omega_q/2) I ⊗ Z + g (a^dag ⊗ sigma_- + a ⊗ sigma_+)``."""
    from eigenlab.pauli import I, Z

    dim = _validate_dim(d, max_dim=128)
    a = annihilation(dim)
    a_dag = a.conj().T
    num = number(dim)
    i_osc = np.eye(dim, dtype=complex)
    sigma_minus = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    sigma_plus = sigma_minus.conj().T

    return (
        float(omega_c) * np.kron(num, I)
        - 0.5 * float(omega_q) * np.kron(i_osc, Z)
        + float(g) * (np.kron(a_dag, sigma_minus) + np.kron(a, sigma_plus))
    )


def excitation_number(d: int) -> np.ndarray:
    """Return the total excitation operator ``a^dag a ⊗ I + I ⊗ |1><1|``."""
    from eigenlab.pauli import I

    dim = _validate_dim(d, max_dim=128)
    proj_1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
    return np.kron(number(dim), I) + np.kron(np.eye(dim, dtype=complex), proj_1)


__all__ = [
    "annihilation",
    "coherent",
    "creation",
    "excitation_number",
    "fock",
    "jaynes_cummings",
    "number",
]
