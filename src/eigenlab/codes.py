"""Quantum error-correcting stabilizer codes and noise channels."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from eigenlab.pauli import apply_pauli, conjugate_clifford, pauli, pauli_comm, pauli_mult
from eigenlab.states import ONE, PLUS, ZERO, basis, density


def _as_density_matrix(state: np.ndarray) -> tuple[np.ndarray, int]:
    """Convert a ket or density matrix to a (2^n, 2^n) complex array and return (rho, n)."""
    mat = np.asarray(state, dtype=complex)
    if mat.ndim == 1:
        mat = density(mat)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Expected square density matrix or ket, got shape {mat.shape}")
    dim = mat.shape[0]
    if dim < 2 or (dim & (dim - 1)) != 0:
        raise ValueError(f"Dimension {dim} is not a positive power of 2")
    n = dim.bit_length() - 1
    if n > 8:
        raise ValueError(f"Matrices only supported for n <= 8 qubits, got n={n}")
    return mat.copy(), n


def bit_flip_channel(rho: np.ndarray, p: float) -> np.ndarray:
    """Apply independent single-qubit bit-flip channels with probability ``p`` to ``n <= 8`` qubits.

    Each qubit independently receives Kraus operators ``sqrt(1 - p) I`` and ``sqrt(p) X``.
    """
    p_val = float(p)
    if not 0.0 <= p_val <= 1.0:
        raise ValueError(f"Probability p must be in [0, 1], got {p}")
    out, n = _as_density_matrix(rho)
    dim = 1 << n
    basis_indices = np.arange(dim, dtype=complex)
    for q in range(n):
        label = "I" * q + "X" + "I" * (n - 1 - q)
        perm = np.real(apply_pauli(label, basis_indices)).astype(int)
        out = (1.0 - p_val) * out + p_val * out[np.ix_(perm, perm)]
    return out


BIT_FLIP_STABILIZERS: tuple[str, str] = ("ZZI", "IZZ")


def stabilizers_commute(stabilizers: Sequence[str]) -> bool:
    """Return True if all stabilizer labels have the same length and commute pairwise."""
    stabs = list(stabilizers)
    if not stabs:
        raise ValueError("stabilizers must be non-empty")
    n = len(stabs[0])
    for i, s1 in enumerate(stabs):
        if len(s1) != n:
            raise ValueError("All stabilizer labels must have the same qubit count.")
        for s2 in stabs[i + 1 :]:
            if pauli_comm(s1, s2)[0] != 0:
                return False
    return True


def syndrome_projector(stabilizers: Sequence[str], syndrome: Sequence[int]) -> np.ndarray:
    """Build the projector Pi_k (I + s_k S_k) / 2 for stabilizer signs s_k in {+1, -1}."""
    stabs = list(stabilizers)
    signs = list(syndrome)
    if len(stabs) != len(signs):
        raise ValueError("stabilizers and syndrome must have the same length.")
    if not stabilizers_commute(stabs):
        raise ValueError("Stabilizers must commute pairwise.")
    n = len(stabs[0])
    if n > 8:
        raise ValueError(f"Matrices only supported for n <= 8 qubits, got n={n}")
    dim = 1 << n
    ident = np.eye(dim, dtype=complex)
    proj = ident.copy()
    for s_label, sign in zip(stabs, signs):
        if sign not in (-1, 1):
            raise ValueError(f"Syndrome sign must be +1 or -1, got {sign}")
        proj = proj @ (0.5 * (ident + float(sign) * pauli(s_label)))
    return proj


def code_projector(stabilizers: Sequence[str]) -> np.ndarray:
    """Build the code-space projector Pi_{S in stabilizers} (I + S) / 2."""
    stabs = list(stabilizers)
    return syndrome_projector(stabs, [1] * len(stabs))


def error_syndrome(error: str, stabilizers: Sequence[str]) -> tuple[int, ...]:
    """Return the syndrome tuple of +1 (commutes) or -1 (anticommutes) for ``error`` against ``stabilizers``."""
    return tuple(1 if pauli_comm(error, s)[0] == 0 else -1 for s in stabilizers)


def encode_bit_flip(state_1q: np.ndarray) -> np.ndarray:
    """Encode a 1-qubit ket alpha|0> + beta|1> into alpha|000> + beta|111>."""
    ket = np.asarray(state_1q, dtype=complex).reshape(-1)
    if ket.shape != (2,):
        raise ValueError(f"Expected 1-qubit ket of shape (2,), got {ket.shape}")
    norm = np.linalg.norm(ket)
    if norm == 0:
        raise ValueError("state_1q has zero norm")
    ket = ket / norm
    return ket[0] * basis(0b000, 3) + ket[1] * basis(0b111, 3)


def correct_bit_flip(rho: np.ndarray) -> np.ndarray:
    """Measure the bit-flip code syndrome (ZZI, IZZ) and apply single-qubit majority recovery."""
    mat, n = _as_density_matrix(rho)
    if n != 3:
        raise ValueError(f"Bit-flip code requires 3 qubits, got {n}")
    recovery_map = {
        (1, 1): "III",
        (-1, 1): "XII",
        (-1, -1): "IXI",
        (1, -1): "IIX",
    }
    out = np.zeros_like(mat)
    for syn, rec_label in recovery_map.items():
        proj = syndrome_projector(BIT_FLIP_STABILIZERS, syn)
        r_mat = pauli(rec_label)
        sub = proj @ mat @ proj
        out += r_mat @ sub @ r_mat.conj().T
    return out


def transversal_clifford(label: str, gate: str) -> tuple[complex, str]:
    """Conjugate ``label`` by single-qubit Clifford ``gate`` on every qubit via ``conjugate_clifford``."""
    phase = 1.0 + 0.0j
    cur = label
    for q in range(len(label)):
        p_q, cur = conjugate_clifford(cur, gate, q)
        phase *= p_q
    return phase, cur


def phase_flip_stabilizers() -> tuple[str, str]:
    """Return the 3-qubit phase-flip code stabilizers obtained by transversal H on the bit-flip code."""
    return tuple(transversal_clifford(s, "H")[1] for s in BIT_FLIP_STABILIZERS)  # type: ignore[return-value]


PHASE_FLIP_STABILIZERS: tuple[str, str] = phase_flip_stabilizers()


def stabilizer_group(generators: Sequence[str]) -> set[str]:
    """Return all Pauli labels in the abelian group generated by ``generators`` via ``pauli_mult``."""
    gens = list(generators)
    if not gens:
        raise ValueError("generators must be non-empty")
    n = len(gens[0])
    group: set[str] = {"I" * n}
    for g in gens:
        next_group = set(group)
        for elem in group:
            _, prod = pauli_mult(elem, g)
            next_group.add(prod)
        group = next_group
    return group


def phase_flip_channel(rho: np.ndarray, p: float) -> np.ndarray:
    """Apply independent single-qubit phase-flip (Z) channels with probability ``p`` to ``n <= 8`` qubits."""
    p_val = float(p)
    if not 0.0 <= p_val <= 1.0:
        raise ValueError(f"Probability p must be in [0, 1], got {p}")
    out, n = _as_density_matrix(rho)
    dim = 1 << n
    eye_cols = np.eye(dim, dtype=complex)
    for q in range(n):
        label = "I" * q + "Z" + "I" * (n - 1 - q)
        # Diagonal signs of Z_q from apply_pauli on the basis
        signs = np.real(np.diag([apply_pauli(label, eye_cols[:, i])[i] for i in range(dim)]))
        z_rho_z = signs @ out @ signs
        out = (1.0 - p_val) * out + p_val * z_rho_z
    return out


def encode_phase_flip(state_1q: np.ndarray) -> np.ndarray:
    """Encode a 1-qubit ket alpha|0> + beta|1> into alpha|+++> + beta|--->."""
    ket = np.asarray(state_1q, dtype=complex).reshape(-1)
    if ket.shape != (2,):
        raise ValueError(f"Expected 1-qubit ket of shape (2,), got {ket.shape}")
    norm = np.linalg.norm(ket)
    if norm == 0:
        raise ValueError("state_1q has zero norm")
    ket = ket / norm
    minus = (ZERO - ONE) / np.sqrt(2.0)
    plus_3 = np.kron(np.kron(PLUS, PLUS), PLUS)
    minus_3 = np.kron(np.kron(minus, minus), minus)
    return ket[0] * plus_3 + ket[1] * minus_3


def correct_phase_flip(rho: np.ndarray) -> np.ndarray:
    """Measure the phase-flip code syndrome (XXI, IXX) and apply single-qubit Z majority recovery."""
    mat, n = _as_density_matrix(rho)
    if n != 3:
        raise ValueError(f"Phase-flip code requires 3 qubits, got {n}")
    recovery_map = {
        (1, 1): "III",
        (-1, 1): "ZII",
        (-1, -1): "IZI",
        (1, -1): "IIZ",
    }
    out = np.zeros_like(mat)
    for syn, rec_label in recovery_map.items():
        proj = syndrome_projector(PHASE_FLIP_STABILIZERS, syn)
        r_mat = pauli(rec_label)
        sub = proj @ mat @ proj
        out += r_mat @ sub @ r_mat.conj().T
    return out


__all__ = [
    "BIT_FLIP_STABILIZERS",
    "PHASE_FLIP_STABILIZERS",
    "apply_pauli",
    "bit_flip_channel",
    "code_projector",
    "correct_bit_flip",
    "correct_phase_flip",
    "encode_bit_flip",
    "encode_phase_flip",
    "error_syndrome",
    "pauli_comm",
    "pauli_mult",
    "phase_flip_channel",
    "phase_flip_stabilizers",
    "stabilizer_group",
    "stabilizers_commute",
    "syndrome_projector",
    "transversal_clifford",
]
