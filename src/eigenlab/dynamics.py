"""Time evolution and Trotter approximations."""

from __future__ import annotations

import numpy as np

from eigenlab.hamiltonian import Hamiltonian, eigensystem
from eigenlab.states import cyclic_shift, grover_once, qft_matrix


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


def trotter_step(
    H: Hamiltonian | Sequence[tuple[complex | float, str]],
    state: np.ndarray,
    dt: float,
) -> np.ndarray:
    """One first-order Trotter step |ψ'⟩ = ∏ exp(-i c_k P_k Δt) |ψ⟩.

    Uses P_k² = I, so each factor is cos(θ) I - i sin(θ) P_k with θ = c_k Δt.
    """
    from eigenlab.pauli import pauli

    terms = H.terms if isinstance(H, Hamiltonian) else list(H)
    ket = np.asarray(state, dtype=complex).reshape(-1).copy()
    for coeff, label in terms:
        theta = complex(coeff) * float(dt)
        p_mat = pauli(label)
        ket = np.cos(theta) * ket - 1j * np.sin(theta) * (p_mat @ ket)
    return ket


def trotter_unitary(
    H: Hamiltonian | Sequence[tuple[complex | float, str]],
    dt: float,
) -> np.ndarray:
    """One first-order Trotter unitary U = ∏ exp(-i c_k P_k Δt)."""
    from eigenlab.pauli import pauli

    terms = H.terms if isinstance(H, Hamiltonian) else list(H)
    if not terms:
        raise ValueError("term list cannot be empty")
    qubits = len(terms[0][1])
    dim = 1 << qubits
    U = np.eye(dim, dtype=complex)
    for coeff, label in terms:
        theta = complex(coeff) * float(dt)
        p_mat = pauli(label)
        factor = np.cos(theta) * np.eye(dim, dtype=complex) - 1j * np.sin(theta) * p_mat
        U = factor @ U
    return U


def strang_step(
    H: Hamiltonian | Sequence[tuple[complex | float, str]],
    state: np.ndarray,
    dt: float,
) -> np.ndarray:
    """One second-order Strang splitting (symmetric Trotter) step on a state ket."""
    from eigenlab.pauli import pauli

    terms = H.terms if isinstance(H, Hamiltonian) else list(H)
    ket = np.asarray(state, dtype=complex).reshape(-1).copy()
    if len(terms) <= 1:
        return trotter_step(terms, ket, dt)
    for coeff, label in terms[:-1]:
        theta = complex(coeff) * (float(dt) / 2.0)
        p_mat = pauli(label)
        ket = np.cos(theta) * ket - 1j * np.sin(theta) * (p_mat @ ket)
    c_last, l_last = terms[-1]
    theta = complex(c_last) * float(dt)
    p_mat = pauli(l_last)
    ket = np.cos(theta) * ket - 1j * np.sin(theta) * (p_mat @ ket)
    for coeff, label in reversed(terms[:-1]):
        theta = complex(coeff) * (float(dt) / 2.0)
        p_mat = pauli(label)
        ket = np.cos(theta) * ket - 1j * np.sin(theta) * (p_mat @ ket)
    return ket


def strang_unitary(
    H: Hamiltonian | Sequence[tuple[complex | float, str]],
    dt: float,
) -> np.ndarray:
    """One second-order Strang splitting (symmetric Trotter) unitary matrix."""
    from eigenlab.pauli import pauli

    terms = H.terms if isinstance(H, Hamiltonian) else list(H)
    if not terms:
        raise ValueError("term list cannot be empty")
    qubits = len(terms[0][1])
    dim = 1 << qubits
    if len(terms) <= 1:
        return trotter_unitary(terms, dt)
    U = np.eye(dim, dtype=complex)
    for coeff, label in terms[:-1]:
        theta = complex(coeff) * (float(dt) / 2.0)
        p_mat = pauli(label)
        factor = np.cos(theta) * np.eye(dim, dtype=complex) - 1j * np.sin(theta) * p_mat
        U = factor @ U
    c_last, l_last = terms[-1]
    theta = complex(c_last) * float(dt)
    p_mat = pauli(l_last)
    factor = np.cos(theta) * np.eye(dim, dtype=complex) - 1j * np.sin(theta) * p_mat
    U = factor @ U
    for coeff, label in reversed(terms[:-1]):
        theta = complex(coeff) * (float(dt) / 2.0)
        p_mat = pauli(label)
        factor = np.cos(theta) * np.eye(dim, dtype=complex) - 1j * np.sin(theta) * p_mat
        U = factor @ U
    return U


def unitary(
    H: np.ndarray | Hamiltonian,
    time: float,
) -> np.ndarray:
    """Exact unitary time-evolution operator U(t) = exp(-i H t)."""
    energies, V = eigensystem(H)
    phases = np.exp(-1j * energies * float(time))
    return V @ np.diag(phases) @ V.conj().T


def loschmidt(
    H: np.ndarray | Hamiltonian,
    state: np.ndarray,
    time: float,
) -> float:
    """Loschmidt echo |⟨ψ(0)|ψ(t)⟩|² with |ψ(t)⟩ from `evolve`."""
    ket = np.asarray(state, dtype=complex).reshape(-1)
    norm = np.linalg.norm(ket)
    if norm == 0:
        raise ValueError("state has zero norm")
    ket = ket / norm
    return float(abs(np.vdot(ket, evolve(H, ket, time))) ** 2)


def spin_echo(
    H: np.ndarray | Hamiltonian,
    state: np.ndarray,
    tau: float,
    pulse: str | None = None,
) -> np.ndarray:
    """Evolve for τ, apply the Pauli `pulse` (default X on every qubit), and evolve for τ again."""
    from eigenlab.pauli import apply_pauli

    ket = np.asarray(state, dtype=complex).reshape(-1)
    if pulse is None:
        pulse = "X" * (ket.size.bit_length() - 1)
    return evolve(H, apply_pauli(pulse, evolve(H, ket, tau)), tau)


def yoshida_unitary(
    H: Hamiltonian | Sequence[tuple[complex | float, str]],
    dt: float,
) -> np.ndarray:
    """Fourth-order Yoshida product S4(Δt) = S2(θΔt) S2((1 − 2θ)Δt) S2(θΔt), θ = 1 / (2 − 2^{1/3}).

    S2 is `strang_unitary`. The middle step (1 − 2θ)Δt is negative.
    """
    theta = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
    outer = strang_unitary(H, theta * float(dt))
    middle = strang_unitary(H, (1.0 - 2.0 * theta) * float(dt))
    return outer @ middle @ outer


def evolve_driven(
    hamiltonian_at,
    state: np.ndarray,
    time: float,
    steps: int,
) -> np.ndarray:
    """Midpoint time-dependent evolution over ``steps`` intervals of length ``dt = time / steps``.

    Each step ``k`` applies ``exp(-i H(t_k + dt/2) dt)`` built with ``eigensystem``.
    """
    if int(steps) < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    n_steps = int(steps)
    dt = float(time) / float(n_steps)
    ket = np.asarray(state, dtype=complex).reshape(-1).copy()
    for k in range(n_steps):
        t_mid = (k + 0.5) * dt
        energies, V = eigensystem(hamiltonian_at(t_mid))
        phases = np.exp(-1j * energies * dt)
        ket = V @ (phases * (V.conj().T @ ket))
    return ket


def expm(A: np.ndarray) -> np.ndarray:
    """Matrix exponential ``exp(A)`` via scaling and squaring of a truncated Taylor series."""
    mat = np.asarray(A, dtype=complex)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"expm requires a square 2D matrix, got shape {mat.shape}")
    n = mat.shape[0]
    norm = float(np.linalg.norm(mat, ord=1))
    s = max(0, int(np.ceil(np.log2(norm / 0.25)))) if norm > 0.25 else 0
    scaled = mat / float(1 << s)

    out = np.eye(n, dtype=complex)
    term = np.eye(n, dtype=complex)
    for k in range(1, 25):
        term = (term @ scaled) / float(k)
        out = out + term
        if float(np.max(np.abs(term))) <= 1e-17:
            break

    for _ in range(s):
        out = out @ out
    return out


def lindbladian(H, jumps):
    from eigenlab.channels import lindbladian as _lindbladian

    return _lindbladian(H, jumps)


def lindblad_evolve(H, jumps, rho: np.ndarray, time: float) -> np.ndarray:
    from eigenlab.channels import lindblad_evolve as _lindblad_evolve

    return _lindblad_evolve(H, jumps, rho, time)


def lindblad_steady_state(H, jumps) -> np.ndarray:
    from eigenlab.channels import lindblad_steady_state as _lindblad_steady_state

    return _lindblad_steady_state(H, jumps)




