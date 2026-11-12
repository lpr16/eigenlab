"""Quantum error-correcting stabilizer codes and noise channels."""

from __future__ import annotations

from collections.abc import Sequence
import itertools

import numpy as np

from eigenlab.hamiltonian import Hamiltonian
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


def stabilizers_commute(stabilizers: Sequence[str] | tuple[Sequence[str], ...]) -> bool:
    """Return True if all stabilizer labels have the same length and commute pairwise."""
    raw = list(stabilizers)
    if not raw:
        raise ValueError("stabilizers must be non-empty")
    if not isinstance(raw[0], str):
        stabs: list[str] = [s for group in raw for s in group]  # type: ignore[union-attr]
    else:
        stabs = raw  # type: ignore[assignment]
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


FIVE_QUBIT_STABILIZERS: tuple[str, str, str, str] = ("XZZXI", "IXZZX", "XIXZZ", "ZXIXZ")


def weight_k_paulis(n: int, k: int) -> list[str]:
    """Return all n-qubit Pauli labels of weight ``k`` (number of non-I characters)."""
    if not 0 <= k <= n:
        raise ValueError(f"Weight k={k} must satisfy 0 <= k <= n={n}")
    if k == 0:
        return ["I" * n]
    labels: list[str] = []
    for positions in itertools.combinations(range(n), k):
        for chars in itertools.product(("X", "Y", "Z"), repeat=k):
            arr = ["I"] * n
            for pos, ch in zip(positions, chars):
                arr[pos] = ch
            labels.append("".join(arr))
    return labels


def single_qubit_paulis(n: int) -> list[str]:
    """Return all 3n single-qubit Pauli labels on n qubits."""
    return weight_k_paulis(n, 1)


def code_distance(generators: Sequence[str]) -> int:
    """Return the minimum weight of a Pauli operator in N(S) \\ S for stabilizer ``generators``."""
    gens = list(generators)
    n = len(gens[0])
    group = stabilizer_group(gens)
    all_ones = (1,) * len(gens)
    for k in range(1, n + 1):
        for lbl in weight_k_paulis(n, k):
            if lbl not in group and error_syndrome(lbl, gens) == all_ones:
                return k
    raise ValueError("No nontrivial logical operators found.")


def knill_laflamme(
    projector_or_stabilizers: np.ndarray | Sequence[str],
    errors: Sequence[str | np.ndarray],
    tol: float = 1e-10,
) -> tuple[bool, np.ndarray]:
    """Check the Knill-Laflamme condition ``P E_a^dag E_b P = c_ab P`` with ``c_ab = Tr(P E_a^dag E_b P) / 2``.

    Returns ``(satisfied, c_matrix)``.
    """
    if isinstance(projector_or_stabilizers, np.ndarray):
        proj = np.asarray(projector_or_stabilizers, dtype=complex)
    else:
        proj = code_projector(projector_or_stabilizers)

    err_mats = [
        pauli(e) if isinstance(e, str) else np.asarray(e, dtype=complex)
        for e in errors
    ]
    pe_mats = [e_mat @ proj for e_mat in err_mats]
    m = len(err_mats)
    c_mat = np.zeros((m, m), dtype=complex)
    satisfied = True
    for a in range(m):
        pe_a_dag = pe_mats[a].conj().T
        for b in range(m):
            block = pe_a_dag @ pe_mats[b]
            c_ab = np.trace(block) / 2.0
            c_mat[a, b] = c_ab
            if np.linalg.norm(block - c_ab * proj) > tol:
                satisfied = False
    return satisfied, c_mat


STEANE_HAMMING_ROWS: tuple[str, str, str] = ("0001111", "0110011", "1010101")
STEANE_X_GENERATORS: tuple[str, str, str] = tuple(
    "".join("X" if bit == "1" else "I" for bit in row) for row in STEANE_HAMMING_ROWS
)  # type: ignore[assignment]
STEANE_Z_GENERATORS: tuple[str, str, str] = tuple(
    "".join("Z" if bit == "1" else "I" for bit in row) for row in STEANE_HAMMING_ROWS
)  # type: ignore[assignment]
STEANE_STABILIZERS: tuple[str, ...] = STEANE_X_GENERATORS + STEANE_Z_GENERATORS


def steane_logical_states() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(|0_L>, |1_L>, |+_L>)`` for the 7-qubit Steane code."""
    ket = basis(0, 7)
    for g_x in STEANE_X_GENERATORS:
        ket = (ket + apply_pauli(g_x, ket)) / np.sqrt(2.0)
    zero_L = ket / np.linalg.norm(ket)
    one_L = apply_pauli("XXXXXXX", zero_L)
    plus_L = (zero_L + one_L) / np.sqrt(2.0)
    return zero_L, one_L, plus_L


def _toric_h(x: int, y: int, L: int = 2) -> int:
    return 2 * (L * (y % L) + (x % L))


def _toric_v(x: int, y: int, L: int = 2) -> int:
    return 2 * (L * (y % L) + (x % L)) + 1


def toric_code_stabilizers(L: int = 2) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ``(stars, plaquettes)`` for the LxL toric code on 2*L^2 qubits."""
    h = lambda x, y: _toric_h(x, y, L)
    v = lambda x, y: _toric_v(x, y, L)


    num_qubits = 2 * L * L
    coords = [(x, y) for x in range(L) for y in range(L)]
    stars: list[str] = []
    plaquettes: list[str] = []
    for x, y in coords:
        star_edges = {h(x, y), h(x - 1, y), v(x, y), v(x, y - 1)}
        stars.append("".join("X" if i in star_edges else "I" for i in range(num_qubits)))
        plaq_edges = {h(x, y), h(x, y + 1), v(x, y), v(x + 1, y)}
        plaquettes.append("".join("Z" if i in plaq_edges else "I" for i in range(num_qubits)))
    return tuple(stars), tuple(plaquettes)



TORIC_STAR_STABILIZERS, TORIC_PLAQUETTE_STABILIZERS = toric_code_stabilizers()
TORIC_CODE_STABILIZERS: tuple[str, ...] = TORIC_STAR_STABILIZERS + TORIC_PLAQUETTE_STABILIZERS


def toric_code_hamiltonian() -> Hamiltonian:
    """Return the 2x2 toric code Hamiltonian ``-sum_s A_s - sum_p B_p`` on 8 qubits."""
    stars, plaquettes = toric_code_stabilizers()
    return Hamiltonian([(-1.0, label) for label in stars + plaquettes])


def toric_code_logicals() -> tuple[str, str, str, str]:
    """Return ``(Z1_bar, X1_bar, Z2_bar, X2_bar)`` logical loop operators on the 2x2 torus."""
    z1_edges = {_toric_h(0, 0), _toric_h(1, 0)}
    x1_edges = {_toric_h(0, 0), _toric_h(0, 1)}
    z2_edges = {_toric_v(0, 0), _toric_v(0, 1)}
    x2_edges = {_toric_v(0, 0), _toric_v(1, 0)}
    z1 = "".join("Z" if i in z1_edges else "I" for i in range(8))
    x1 = "".join("X" if i in x1_edges else "I" for i in range(8))
    z2 = "".join("Z" if i in z2_edges else "I" for i in range(8))
    x2 = "".join("X" if i in x2_edges else "I" for i in range(8))
    return z1, x1, z2, x2


TORIC_LOGICAL_OPERATORS: tuple[str, str, str, str] = toric_code_logicals()


__all__ = [
    "BIT_FLIP_STABILIZERS",
    "FIVE_QUBIT_STABILIZERS",
    "PHASE_FLIP_STABILIZERS",
    "STEANE_HAMMING_ROWS",
    "STEANE_STABILIZERS",
    "STEANE_X_GENERATORS",
    "STEANE_Z_GENERATORS",
    "TORIC_CODE_STABILIZERS",
    "TORIC_LOGICAL_OPERATORS",
    "TORIC_PLAQUETTE_STABILIZERS",
    "TORIC_STAR_STABILIZERS",
    "apply_pauli",
    "bit_flip_channel",
    "code_distance",
    "code_projector",
    "correct_bit_flip",
    "correct_phase_flip",
    "encode_bit_flip",
    "encode_phase_flip",
    "error_syndrome",
    "knill_laflamme",
    "pauli_comm",
    "pauli_mult",
    "phase_flip_channel",
    "phase_flip_stabilizers",
    "single_qubit_paulis",
    "stabilizer_group",
    "stabilizers_commute",
    "steane_logical_states",
    "syndrome_projector",
    "toric_code_hamiltonian",
    "toric_code_logicals",
    "toric_code_stabilizers",
    "transversal_clifford",
    "weight_k_paulis",
]
