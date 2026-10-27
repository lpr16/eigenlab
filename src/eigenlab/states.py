"""Computational-basis kets and two-qubit Bell states."""

from __future__ import annotations

import numpy as np

ZERO = np.array([1, 0], dtype=complex)
ONE = np.array([0, 1], dtype=complex)
PLUS = np.array([1, 1], dtype=complex) / np.sqrt(2)
PLUS_I = np.array([1, 1j], dtype=complex) / np.sqrt(2)


def bloch(state: np.ndarray) -> np.ndarray:
    """Bloch vector (r_x, r_y, r_z) of a one-qubit ket or 2 x 2 density matrix.

    Pure states lie on the unit sphere S^2, mixed states inside it. Multi-qubit states are rejected.
    """
    arr = np.asarray(state, dtype=complex)
    if arr.shape == (2, 2):
        tr = np.trace(arr)
        if abs(tr) == 0:
            raise ValueError("density matrix has zero trace")
        rho = arr / tr
        # r = (Tr ρX, Tr ρY, Tr ρZ) = (2 Re ρ₁₀, 2 Im ρ₁₀, ρ₀₀ − ρ₁₁)
        return np.array(
            [2.0 * rho[1, 0].real, 2.0 * rho[1, 0].imag, (rho[0, 0] - rho[1, 1]).real],
            dtype=float,
        )
    ket = arr.reshape(-1)
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


def ghz(n: int) -> np.ndarray:
    """n-qubit GHZ state (|0...0⟩ + |1...1⟩) / √2 for 2 <= n <= 8."""
    if not 2 <= n <= 8:
        raise ValueError("ghz requires 2 <= n <= 8 qubits")
    dim = 1 << n
    ket = np.zeros(dim, dtype=complex)
    ket[0] = 1.0 / np.sqrt(2.0)
    ket[dim - 1] = 1.0 / np.sqrt(2.0)
    return ket


def w_state(n: int = 3) -> np.ndarray:
    """Three-qubit W state (|001⟩ + |010⟩ + |100⟩) / √3 (or n-qubit generalization)."""
    if not 2 <= n <= 8:
        raise ValueError("w_state requires 2 <= n <= 8 qubits")
    if n == 3:
        return (basis(1, 3) + basis(2, 3) + basis(4, 3)) / np.sqrt(3.0)
    ket = np.zeros(1 << n, dtype=complex)
    for k in range(n):
        ket[1 << k] = 1.0 / np.sqrt(float(n))
    return ket


def purify(rho: np.ndarray) -> np.ndarray:
    """Purify a one-qubit density matrix ρ = ∑_i p_i |u_i⟩⟨u_i| into ∑_i √p_i |u_i⟩ ⊗ |i⟩."""
    mat = np.asarray(rho, dtype=complex)
    if mat.shape != (2, 2):
        raise ValueError("purify requires a 2x2 one-qubit density matrix")
    evals, evecs = np.linalg.eigh(mat)
    idx = np.argsort(np.real(evals))[::-1]
    evals = np.maximum(0.0, np.real(evals[idx]))
    evecs = evecs[:, idx]
    ket = np.zeros(4, dtype=complex)
    for i in range(2):
        ket += np.sqrt(evals[i]) * np.kron(evecs[:, i], basis(i, 1))
    return ket


def negativity(state: np.ndarray) -> float:
    """Negativity of a two-qubit ket or density matrix (sum of |λ_i| for λ_i < 0 of ρ^{T_1})."""
    arr = np.asarray(state, dtype=complex)
    if arr.ndim == 1 or (arr.ndim == 2 and (arr.shape[0] == 1 or arr.shape[1] == 1)):
        ket = arr.reshape(-1)
        if ket.size != 4:
            raise ValueError("negativity requires a two-qubit ket (size 4) or 4x4 density matrix")
        rho = density(ket)
    elif arr.ndim == 2 and arr.shape == (4, 4):
        tr = np.trace(arr)
        if abs(tr) == 0:
            raise ValueError("density matrix has zero trace")
        rho = arr / tr
    else:
        raise ValueError("negativity requires a two-qubit ket (size 4) or 4x4 density matrix")

    rho_pt = rho.reshape((2, 2, 2, 2)).transpose((0, 3, 2, 1)).reshape((4, 4))
    evals = np.linalg.eigvalsh(rho_pt)
    neg_evals = evals[evals < -1e-14]
    val = float(-np.sum(neg_evals))
    if val < 1e-14:
        return 0.0
    return val


def entanglement_of_formation(state: np.ndarray) -> float:
    """Entanglement of formation in bits of a two-qubit state from its concurrence C."""
    c = concurrence(state)
    x = 0.5 * (1.0 + np.sqrt(max(0.0, 1.0 - c * c)))
    if x <= 1e-15 or x >= 1.0 - 1e-15:
        return 0.0
    val = float(-x * np.log2(x) - (1.0 - x) * np.log2(1.0 - x))
    if abs(val) < 1e-14:
        return 0.0
    return val


from eigenlab.pauli import pauli_probabilities, post_measure


def probabilities(state: np.ndarray) -> np.ndarray:
    ket = np.asarray(state, dtype=complex).reshape(-1)
    norm = np.linalg.norm(ket)
    if norm == 0:
        raise ValueError("state has zero norm")
    return np.abs(ket / norm) ** 2


def shannon_entropy(state: np.ndarray) -> float:
    """Shannon entropy in bits of a computational-basis measurement on `state`."""
    probs = probabilities(state)
    probs = probs[probs > 1e-15]
    ent = float(-np.sum(probs * np.log2(probs)))
    if abs(ent) < 1e-14:
        return 0.0
    return ent


def helstrom(psi: np.ndarray, phi: np.ndarray) -> float:
    """Helstrom minimum-error success probability (1 + √(1 - |⟨ψ|φ⟩|²)) / 2 for equal priors."""
    v1 = np.asarray(psi, dtype=complex).reshape(-1)
    v2 = np.asarray(phi, dtype=complex).reshape(-1)
    if v1.size != v2.size:
        raise ValueError("states must have the same dimension")
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        raise ValueError("state has zero norm")
    v1 = v1 / n1
    v2 = v2 / n2
    overlap_sq = float(abs(np.vdot(v1, v2)) ** 2)
    return float(0.5 * (1.0 + np.sqrt(max(0.0, 1.0 - overlap_sq))))


def swap_test(psi: np.ndarray, phi: np.ndarray) -> float:
    """Three-qubit swap test circuit returning P(ancilla = 0) = (1 + |⟨ψ|φ⟩|²) / 2."""
    v1 = np.asarray(psi, dtype=complex).reshape(-1)
    v2 = np.asarray(phi, dtype=complex).reshape(-1)
    if v1.size != 2 or v2.size != 2:
        raise ValueError("swap_test requires two one-qubit kets of size 2")
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        raise ValueError("state has zero norm")
    v1 = v1 / n1
    v2 = v2 / n2

    # Initial 3-qubit state |0⟩ ⊗ |ψ⟩ ⊗ |φ⟩
    state0 = np.kron(np.kron(ZERO, v1), v2)

    # H on ancilla (qubit 0)
    h_1q = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    h_anc = np.kron(h_1q, np.eye(4, dtype=complex))

    # Fredkin gate: swaps qubits 1 and 2 when qubit 0 is 1 (swaps |101⟩ (5) and |110⟩ (6))
    fredkin = np.eye(8, dtype=complex)
    fredkin[5, 5] = 0.0
    fredkin[6, 6] = 0.0
    fredkin[5, 6] = 1.0
    fredkin[6, 5] = 1.0

    final_state = h_anc @ (fredkin @ (h_anc @ state0))
    prob_anc_0 = float(np.sum(np.abs(final_state[:4]) ** 2))
    return prob_anc_0


def grover_once(marked: int, qubits: int = 2) -> np.ndarray:
    """One Grover iteration (2|s⟩⟨s| - I) O |s⟩ on `qubits` qubits with marked basis index `marked`."""
    if not 1 <= qubits <= 8:
        raise ValueError("grover_once requires 1 <= qubits <= 8")
    dim = 1 << qubits
    if not 0 <= marked < dim:
        raise ValueError(f"marked index {marked} out of range for {qubits} qubits")
    s = np.ones(dim, dtype=complex) / np.sqrt(float(dim))
    oracle = np.eye(dim, dtype=complex)
    oracle[marked, marked] = -1.0
    diffuser = 2.0 * np.outer(s, np.conj(s)) - np.eye(dim, dtype=complex)
    return diffuser @ (oracle @ s)


def qft_matrix(n: int) -> np.ndarray:
    """Quantum Fourier transform matrix on `n` qubits (1 <= n <= 8): Q_{j,k} = exp(2πi jk / 2^n) / √2^n."""
    if not 1 <= n <= 8:
        raise ValueError("qft_matrix requires 1 <= n <= 8 qubits")
    dim = 1 << n
    j = np.arange(dim, dtype=float)[:, None]
    k = np.arange(dim, dtype=float)[None, :]
    return np.exp(2.0j * np.pi * (j * k) / float(dim)) / np.sqrt(float(dim))


def cyclic_shift(n: int) -> np.ndarray:
    """Cyclic shift permutation matrix T on `n` qubits with T|k⟩ = |k+1 mod 2^n⟩."""
    if not 1 <= n <= 8:
        raise ValueError("cyclic_shift requires 1 <= n <= 8 qubits")
    dim = 1 << n
    mat = np.zeros((dim, dim), dtype=complex)
    for k in range(dim):
        mat[(k + 1) % dim, k] = 1.0
    return mat


def chsh_operator():
    """CHSH operator A⊗B + A⊗B' + A'⊗B - A'⊗B' with A=Z, A'=X, B=(Z+X)/√2, B'=(Z-X)/√2."""
    from eigenlab.hamiltonian import Hamiltonian

    s = 1.0 / np.sqrt(2.0)
    # A⊗B = (ZZ + ZX)/√2, A⊗B' = (ZZ - ZX)/√2, A'⊗B = (XZ + XX)/√2, -A'⊗B' = (-XZ + XX)/√2
    terms = [
        (s, "ZZ"),
        (s, "ZX"),
        (s, "ZZ"),
        (-s, "ZX"),
        (s, "XZ"),
        (s, "XX"),
        (-s, "XZ"),
        (s, "XX"),
    ]
    return Hamiltonian(terms).simplify()


def relative_entropy(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Quantum relative entropy S(ρ || σ) = Tr(ρ log2 ρ) - Tr(ρ log2 σ) in bits."""
    r = np.asarray(rho, dtype=complex)
    s = np.asarray(sigma, dtype=complex)
    if r.ndim == 1:
        r = density(r)
    if s.ndim == 1:
        s = density(s)
    if r.ndim != 2 or r.shape[0] != r.shape[1] or r.shape != s.shape:
        raise ValueError("rho and sigma must be square matrices of the same dimension")
    tr_r = np.trace(r)
    tr_s = np.trace(s)
    if abs(tr_r) == 0 or abs(tr_s) == 0:
        raise ValueError("density matrix has zero trace")
    r = r / tr_r
    s = s / tr_s

    evals_s, evecs_s = np.linalg.eigh(s)
    if np.any(np.real(evals_s) <= 1e-12):
        raise ValueError("sigma must be positive definite")

    log2_s = evecs_s @ np.diag(np.log2(np.real(evals_s))) @ evecs_s.conj().T
    evals_r = np.real(np.linalg.eigvalsh(r))
    pos_r = evals_r[evals_r > 1e-15]
    tr_r_log2_r = float(np.sum(pos_r * np.log2(pos_r))) if pos_r.size else 0.0
    tr_r_log2_s = float(np.real(np.trace(r @ log2_s)))
    val = tr_r_log2_r - tr_r_log2_s
    if abs(val) < 1e-14:
        return 0.0
    return float(val)
