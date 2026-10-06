"""Computational-basis kets and two-qubit Bell states."""

from __future__ import annotations

import numpy as np

ZERO = np.array([1, 0], dtype=complex)
ONE = np.array([0, 1], dtype=complex)
PLUS = np.array([1, 1], dtype=complex) / np.sqrt(2)
PLUS_I = np.array([1, 1j], dtype=complex) / np.sqrt(2)


def bloch(state: np.ndarray) -> np.ndarray:
    """Bloch vector (r_x, r_y, r_z) of a one-qubit ket.

    Pure states lie on the unit sphere S^2. Multi-qubit states are rejected.
    """
    ket = np.asarray(state, dtype=complex).reshape(-1)
    if ket.size != 2:
        raise ValueError("state must be a one-qubit ket")
    norm = np.linalg.norm(ket)
    if norm == 0:
        raise ValueError("state has zero norm")
    ket = ket / norm
    rx = 2.0 * float((np.conj(ket[0]) * ket[1]).real)
    ry = 2.0 * float((np.conj(ket[0]) * ket[1]).imag)
    rz = float((np.abs(ket[0]) ** 2 - np.abs(ket[1]) ** 2).real)
    return np.array([rx, ry, rz], dtype=float)


def basis(index: int, qubits: int) -> np.ndarray:
    """Computational-basis ket |index⟩ on `qubits` qubits."""
    if qubits < 1 or not 0 <= index < (1 << qubits):
        raise ValueError("index out of range for the given number of qubits")
    ket = np.zeros(1 << qubits, dtype=complex)
    ket[index] = 1
    return ket


def density(state: np.ndarray) -> np.ndarray:
    """Density operator ρ = |ψ⟩⟨ψ| from a ket |ψ⟩."""
    ket = np.asarray(state, dtype=complex).reshape(-1)
    norm = np.linalg.norm(ket)
    if norm == 0:
        raise ValueError("state has zero norm")
    ket = ket / norm
    return np.outer(ket, np.conj(ket))


def purity(rho: np.ndarray) -> float:
    """Purity γ = Tr(ρ²) of a density operator ρ."""
    mat = np.asarray(rho, dtype=complex)
    if mat.ndim == 1:
        mat = density(mat)
    elif mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("rho must be a square matrix or a ket")
    val = np.trace(mat @ mat)
    return float(val.real)


def partial_trace(
    rho: np.ndarray,
    qubits: int,
    keep: int | list[int] | tuple[int, ...] = (),
) -> np.ndarray | float | int:
    """Partial trace of a density operator or ket over unkept qubits.

    qubits: total number of qubits.
    keep: index or collection of qubit indices to retain.
          If empty, traces out all qubits and returns the total trace.
    """
    mat = np.asarray(rho, dtype=complex)
    if mat.ndim == 1:
        mat = density(mat)
    elif mat.ndim != 2 or mat.shape != (1 << qubits, 1 << qubits):
        raise ValueError("rho shape does not match the given number of qubits")

    if isinstance(keep, int):
        keep_qubits = [keep]
    else:
        keep_qubits = list(keep)

    for q in keep_qubits:
        if not 0 <= q < qubits:
            raise ValueError(f"qubit index {q} out of range for {qubits} qubits")
    if len(set(keep_qubits)) != len(keep_qubits):
        raise ValueError("duplicate qubit indices in keep")

    if len(keep_qubits) == 0:
        val = float(np.real(np.trace(mat)))
        if np.isclose(val, round(val), atol=1e-10):
            return int(round(val))
        return val

    tensor = mat.reshape([2] * (2 * qubits))
    in_indices = list(range(2 * qubits))
    for q in range(qubits):
        if q not in keep_qubits:
            in_indices[qubits + q] = q
    out_indices = keep_qubits + [qubits + q for q in keep_qubits]
    res = np.einsum(tensor, in_indices, out_indices)
    k = len(keep_qubits)
    return res.reshape((1 << k, 1 << k))


def schmidt_spectrum(state: np.ndarray) -> np.ndarray:
    """Schmidt spectrum (singular values) across the middle bipartition."""
    ket = np.asarray(state, dtype=complex).reshape(-1)
    dim = ket.size
    qubits = int(np.round(np.log2(dim)))
    if (1 << qubits) != dim or qubits < 2:
        raise ValueError("state must be at least two qubits with dimension 2^n")
    norm = np.linalg.norm(ket)
    if norm == 0:
        raise ValueError("state has zero norm")
    ket = ket / norm
    cut = qubits // 2
    dim_a = 1 << cut
    dim_b = 1 << (qubits - cut)
    mat = ket.reshape((dim_a, dim_b))
    return np.linalg.svd(mat, compute_uv=False)


def entanglement_entropy(state: np.ndarray) -> float:
    """Von Neumann entanglement entropy in bits across the middle cut."""
    s = schmidt_spectrum(state)
    probs = s**2
    probs = probs[probs > 1e-15]
    ent = float(-np.sum(probs * np.log2(probs)))
    if abs(ent) < 1e-14:
        return 0.0
    return ent


def concurrence(state: np.ndarray) -> float:
    """Concurrence of a two-qubit state, lying in [0, 1]."""
    arr = np.asarray(state, dtype=complex)
    if arr.ndim == 1 or (arr.ndim == 2 and (arr.shape[0] == 1 or arr.shape[1] == 1)):
        ket = arr.reshape(-1)
        if ket.size != 4:
            raise ValueError("state must be a two-qubit state (size 4)")
        norm = np.linalg.norm(ket)
        if norm == 0:
            raise ValueError("state has zero norm")
        ket = ket / norm
        val = 2.0 * abs(ket[0] * ket[3] - ket[1] * ket[2])
        if val < 1e-14:
            return 0.0
        return float(np.clip(val, 0.0, 1.0))
    elif arr.ndim == 2 and arr.shape == (4, 4):
        tr = np.trace(arr)
        if abs(tr) == 0:
            raise ValueError("density matrix has zero trace")
        rho = arr / tr
        yy = np.array(
            [[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]],
            dtype=complex,
        )
        rho_tilde = yy @ np.conj(rho) @ yy
        r = rho @ rho_tilde
        evals = np.sort(np.real(np.linalg.eigvals(r)))[::-1]
        lambdas = np.sqrt(np.maximum(0.0, evals))
        c = lambdas[0] - float(np.sum(lambdas[1:]))
        if c < 1e-14:
            return 0.0
        return float(np.clip(c, 0.0, 1.0))
    else:
        raise ValueError("state must be a two-qubit ket (size 4) or density matrix (4x4)")


def bell(kind: str = "phi+") -> np.ndarray:
    """One of the four Bell states."""
    phi_plus = (np.kron(ZERO, ZERO) + np.kron(ONE, ONE)) / np.sqrt(2)
    phi_minus = (np.kron(ZERO, ZERO) - np.kron(ONE, ONE)) / np.sqrt(2)
    psi_plus = (np.kron(ZERO, ONE) + np.kron(ONE, ZERO)) / np.sqrt(2)
    psi_minus = (np.kron(ZERO, ONE) - np.kron(ONE, ZERO)) / np.sqrt(2)
    states = {"phi+": phi_plus, "phi-": phi_minus, "psi+": psi_plus, "psi-": psi_minus}
    try:
        return states[kind]
    except KeyError as exc:
        raise ValueError("kind must be phi+, phi-, psi+, or psi-") from exc


def probabilities(state: np.ndarray) -> np.ndarray:
    ket = np.asarray(state, dtype=complex).reshape(-1)
    norm = np.linalg.norm(ket)
    if norm == 0:
        raise ValueError("state has zero norm")
    return np.abs(ket / norm) ** 2
