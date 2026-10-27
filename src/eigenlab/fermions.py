"""Fermionic creation and annihilation operators via Jordan-Wigner transformation."""

from __future__ import annotations

import numpy as np

from eigenlab.hamiltonian import Hamiltonian


def jordan_wigner_a(p: int, n: int) -> Hamiltonian:
    """Annihilation operator a_p for orbital p out of n fermions via Jordan-Wigner.

    a_p = (Z^⊗p ⊗ (X + iY)/2 ⊗ I^⊗(n-1-p))
    """
    if not 0 <= p < n:
        raise ValueError(f"orbital {p} out of range for {n} fermions")
    label_x = "Z" * p + "X" + "I" * (n - 1 - p)
    label_y = "Z" * p + "Y" + "I" * (n - 1 - p)
    return Hamiltonian([(0.5, label_x), (0.5j, label_y)])


def jordan_wigner_adag(p: int, n: int) -> Hamiltonian:
    """Creation operator a†_p for orbital p out of n fermions via Jordan-Wigner.

    a†_p = (Z^⊗p ⊗ (X - iY)/2 ⊗ I^⊗(n-1-p))
    """
    if not 0 <= p < n:
        raise ValueError(f"orbital {p} out of range for {n} fermions")
    label_x = "Z" * p + "X" + "I" * (n - 1 - p)
    label_y = "Z" * p + "Y" + "I" * (n - 1 - p)
    return Hamiltonian([(0.5, label_x), (-0.5j, label_y)])


def jordan_wigner(p: int, n: int, dagger: bool = False) -> Hamiltonian:
    """Jordan-Wigner operator for orbital p. Returns a†_p if dagger=True, else a_p."""
    if dagger:
        return jordan_wigner_adag(p, n)
    return jordan_wigner_a(p, n)


jordan_wigner_annihilate = jordan_wigner_a
jordan_wigner_create = jordan_wigner_adag


_CNOT_MAP = {
    ("I", "I"): (1.0, "I", "I"),
    ("I", "X"): (1.0, "I", "X"),
    ("I", "Y"): (1.0, "Z", "Y"),
    ("I", "Z"): (1.0, "Z", "Z"),
    ("X", "I"): (1.0, "X", "X"),
    ("X", "X"): (1.0, "X", "I"),
    ("X", "Y"): (1.0, "Y", "Z"),
    ("X", "Z"): (-1.0, "Y", "Y"),
    ("Y", "I"): (1.0, "Y", "X"),
    ("Y", "X"): (1.0, "Y", "I"),
    ("Y", "Y"): (-1.0, "X", "Z"),
    ("Y", "Z"): (1.0, "X", "Y"),
    ("Z", "I"): (1.0, "Z", "I"),
    ("Z", "X"): (1.0, "Z", "X"),
    ("Z", "Y"): (1.0, "I", "Y"),
    ("Z", "Z"): (1.0, "I", "Z"),
}


def _fenwick_cnot_gates(n: int) -> list[tuple[int, int]]:
    """Return sequence of (control, target) CNOT gates transforming occupation basis to Bravyi-Kitaev."""
    gates: list[tuple[int, int]] = []
    for q in range(n):
        i = q + 1
        p = i + (i & -i)
        if p <= n:
            gates.append((q, p - 1))
    return gates


def _conjugate_by_cnots(ham: Hamiltonian, gates: list[tuple[int, int]]) -> Hamiltonian:
    terms = ham.terms
    for c, t in gates:
        new_terms: list[tuple[complex, str]] = []
        for coeff, label in terms:
            chars = list(label)
            phase, nc, nt = _CNOT_MAP[(chars[c], chars[t])]
            chars[c] = nc
            chars[t] = nt
            new_terms.append((coeff * phase, "".join(chars)))
        terms = new_terms
    return Hamiltonian(terms)


def bravyi_kitaev_a(p: int, n: int) -> Hamiltonian:
    """Annihilation operator a_p for orbital p out of n fermions via Bravyi-Kitaev."""
    if not 0 <= p < n:
        raise ValueError(f"orbital {p} out of range for {n} fermions")
    jw = jordan_wigner_a(p, n)
    return _conjugate_by_cnots(jw, _fenwick_cnot_gates(n))


def bravyi_kitaev_adag(p: int, n: int) -> Hamiltonian:
    """Creation operator a†_p for orbital p out of n fermions via Bravyi-Kitaev."""
    if not 0 <= p < n:
        raise ValueError(f"orbital {p} out of range for {n} fermions")
    jw = jordan_wigner_adag(p, n)
    return _conjugate_by_cnots(jw, _fenwick_cnot_gates(n))


def bravyi_kitaev(p: int, n: int, dagger: bool = False) -> Hamiltonian:
    """Bravyi-Kitaev operator for orbital p. Returns a†_p if dagger=True, else a_p."""
    if dagger:
        return bravyi_kitaev_adag(p, n)
    return bravyi_kitaev_a(p, n)


bravyi_kitaev_annihilate = bravyi_kitaev_a
bravyi_kitaev_create = bravyi_kitaev_adag


def _ladder_operators(mapping: str):
    """Return the (annihilation, creation) builders for `mapping`."""
    map_norm = mapping.lower()
    if map_norm in ("jordan_wigner", "jw"):
        return jordan_wigner_a, jordan_wigner_adag
    if map_norm in ("bravyi_kitaev", "bk"):
        return bravyi_kitaev_a, bravyi_kitaev_adag
    raise ValueError(f"Unknown mapping: {mapping}. Must be 'jordan_wigner' or 'bravyi_kitaev'")


def number_operator(p: int, n: int, mapping: str = "jordan_wigner") -> Hamiltonian:
    """Number operator n_p = a†_p a_p."""
    if not 0 <= p < n:
        raise ValueError(f"orbital {p} out of range for {n} fermions")
    map_norm = mapping.lower()
    if map_norm in ("jordan_wigner", "jw"):
        label_i = "I" * n
        label_z = "I" * p + "Z" + "I" * (n - 1 - p)
        return Hamiltonian([(0.5, label_i), (-0.5, label_z)])
    elif map_norm in ("bravyi_kitaev", "bk"):
        return (bravyi_kitaev_adag(p, n) @ bravyi_kitaev_a(p, n)).simplify()
    else:
        raise ValueError(f"Unknown mapping: {mapping}. Must be 'jordan_wigner' or 'bravyi_kitaev'")


def total_number_operator(n: int, mapping: str = "jordan_wigner") -> Hamiltonian:
    """Total number operator N = ∑_{p=0}^{n-1} a†_p a_p."""
    if mapping.lower() not in ("jordan_wigner", "jw"):
        N = Hamiltonian([])
        for p in range(n):
            N = N + number_operator(p, n, mapping=mapping)
        return N.simplify()
    terms: list[tuple[complex | float, str]] = [(0.5 * n, "I" * n)]
    for p in range(n):
        label_z = "I" * p + "Z" + "I" * (n - 1 - p)
        terms.append((-0.5, label_z))
    return Hamiltonian(terms)


def tight_binding_chain(
    n: int, t: float = 1.0, mapping: str = "jordan_wigner"
) -> Hamiltonian:
    """Open tight-binding chain on n spinless fermions: H = -t ∑_i (a†_i a_{i+1} + h.c.)."""
    if n < 2:
        raise ValueError("tight binding chain requires at least 2 sites")
    map_norm = mapping.lower()
    if map_norm in ("jordan_wigner", "jw"):
        terms: list[tuple[complex | float, str]] = []
        for i in range(n - 1):
            # a†_i a_{i+1} + a†_{i+1} a_i = (X_i X_{i+1} + Y_i Y_{i+1}) / 2
            label_xx = ["I"] * n
            label_xx[i] = "X"
            label_xx[i + 1] = "X"
            terms.append((-0.5 * float(t), "".join(label_xx)))

            label_yy = ["I"] * n
            label_yy[i] = "Y"
            label_yy[i + 1] = "Y"
            terms.append((-0.5 * float(t), "".join(label_yy)))
        return Hamiltonian(terms)
    elif map_norm in ("bravyi_kitaev", "bk"):
        H = Hamiltonian([])
        for i in range(n - 1):
            hop = (
                bravyi_kitaev_adag(i, n) @ bravyi_kitaev_a(i + 1, n)
                + bravyi_kitaev_adag(i + 1, n) @ bravyi_kitaev_a(i, n)
            )
            H = H + (-float(t) * hop)
        return H.simplify()
    else:
        raise ValueError(f"Unknown mapping: {mapping}. Must be 'jordan_wigner' or 'bravyi_kitaev'")


def tight_binding_ring(
    n: int, t: float = 1.0, mapping: str = "jordan_wigner"
) -> Hamiltonian:
    """Periodic tight-binding ring: tight_binding_chain plus the closing bond −t (a†_{n−1} a_0 + h.c.)."""
    if n < 3:
        raise ValueError("tight binding ring requires at least 3 sites")
    a_fn, adag_fn = _ladder_operators(mapping)
    closing = adag_fn(n - 1, n) @ a_fn(0, n) + adag_fn(0, n) @ a_fn(n - 1, n)
    return (tight_binding_chain(n, t=t, mapping=mapping) + (-float(t)) * closing).simplify()


def ssh_chain(
    n: int, v: float = 1.0, w: float = 1.0, mapping: str = "jordan_wigner"
) -> Hamiltonian:
    """Open Su-Schrieffer-Heeger (SSH) chain on even `n >= 2` sites with alternating hoppings v, w."""
    if n < 2 or n % 2 != 0:
        raise ValueError("ssh_chain requires an even number of sites n >= 2")
    a_fn, adag_fn = _ladder_operators(mapping)

    H = Hamiltonian([(0.0, "I" * n)])
    for i in range(n - 1):
        amp = float(v) if (i % 2 == 0) else float(w)
        if abs(amp) > 1e-15:
            hop = adag_fn(i, n) @ a_fn(i + 1, n) + adag_fn(i + 1, n) @ a_fn(i, n)
            H = H + (-amp * hop)
    return H.simplify()


def kitaev_chain(
    n: int,
    t: float = 1.0,
    delta: float = 1.0,
    mu: float = 0.0,
    mapping: str = "jordan_wigner",
) -> Hamiltonian:
    """Open Kitaev chain on n spinless fermions.

    H = Σ_i −μ (n_i − 1/2) + Σ_i [−t (a†_i a_{i+1} + h.c.) + Δ (a_i a_{i+1} + a†_{i+1} a†_i)]
    """
    if n < 2:
        raise ValueError("kitaev_chain requires at least 2 sites")
    a_fn, adag_fn = _ladder_operators(mapping)
    identity = Hamiltonian([(1.0, "I" * n)])

    H = Hamiltonian([(0.0, "I" * n)])
    for i in range(n):
        H = H + (-float(mu)) * (adag_fn(i, n) @ a_fn(i, n) - 0.5 * identity)
    for i in range(n - 1):
        hop = adag_fn(i, n) @ a_fn(i + 1, n) + adag_fn(i + 1, n) @ a_fn(i, n)
        pair = a_fn(i, n) @ a_fn(i + 1, n) + adag_fn(i + 1, n) @ adag_fn(i, n)
        H = H + (-float(t)) * hop + float(delta) * pair
    return H.simplify()


def kitaev_bdg_matrix(
    n: int, t: float = 1.0, delta: float = 1.0, mu: float = 0.0
) -> np.ndarray:
    """Bogoliubov–de Gennes 2n x 2n matrix [[A, B], [−B, −A]] for the open Kitaev chain."""
    if n < 2:
        raise ValueError("kitaev_chain requires at least 2 sites")
    A = np.zeros((n, n), dtype=float)
    B = np.zeros((n, n), dtype=float)
    for i in range(n):
        A[i, i] = -float(mu)
    for i in range(n - 1):
        A[i, i + 1] = -float(t)
        A[i + 1, i] = -float(t)
        B[i, i + 1] = float(delta)
        B[i + 1, i] = -float(delta)
    return np.block([[A, B], [-B, -A]])


def kitaev_bogoliubov_spectrum(
    n: int, t: float = 1.0, delta: float = 1.0, mu: float = 0.0
) -> np.ndarray:
    """All 2^n many-body energies Σ_k (ν_k − 1/2) ε_k of the open Kitaev chain, ascending."""
    import itertools

    eps = np.linalg.eigvalsh(kitaev_bdg_matrix(n, t=t, delta=delta, mu=mu))[n:]
    energies = [
        sum((nu_k - 0.5) * eps[k] for k, nu_k in enumerate(nu))
        for nu in itertools.product([0, 1], repeat=n)
    ]
    return np.sort(np.array(energies, dtype=float))


def total_spin_z(n_spatial: int, mapping: str = "jordan_wigner") -> Hamiltonian:
    """Total spin projection S_z = (1/2) ∑_i (n_{i↑} - n_{i↓}).

    Spin-orbitals are interleaved as: orbital 2*i is spin-up, 2*i + 1 is spin-down.
    """
    n_orbitals = 2 * n_spatial
    if mapping.lower() not in ("jordan_wigner", "jw"):
        s_z = Hamiltonian([])
        for i in range(n_spatial):
            s_z = s_z + 0.5 * (
                number_operator(2 * i, n_orbitals, mapping=mapping)
                - number_operator(2 * i + 1, n_orbitals, mapping=mapping)
            )
        return s_z.simplify()
    terms: list[tuple[complex | float, str]] = []
    # n_p = (I - Z_p)/2
    # S_z = (1/2) ∑ ( (I - Z_{2i})/2 - (I - Z_{2i+1})/2 ) = (1/4) ∑ (-Z_{2i} + Z_{2i+1})
    for i in range(n_spatial):
        lbl_up = ["I"] * n_orbitals
        lbl_up[2 * i] = "Z"
        terms.append((-0.25, "".join(lbl_up)))

        lbl_dn = ["I"] * n_orbitals
        lbl_dn[2 * i + 1] = "Z"
        terms.append((0.25, "".join(lbl_dn)))
    return Hamiltonian(terms)


def total_spin_squared(n_spatial: int) -> Hamiltonian:
    """Total spin operator S² = S_- S_+ + S_z² + S_z for interleaved fermions.

    S_+ = ∑_i a†_{2i} a_{2i+1}, S_- = S_+†, and S_z is total_spin_z.
    """
    if n_spatial < 1:
        raise ValueError("n_spatial must be at least 1")
    n_orbitals = 2 * n_spatial
    s_plus = Hamiltonian([])
    for i in range(n_spatial):
        term = jordan_wigner_adag(2 * i, n_orbitals) @ jordan_wigner_a(2 * i + 1, n_orbitals)
        s_plus = s_plus + term
    s_minus = s_plus.dagger()
    s_z = total_spin_z(n_spatial)

    s2 = (s_minus @ s_plus) + (s_z @ s_z) + s_z
    return s2.simplify()


def hubbard_dimer(
    t: float = 1.0, U: float = 0.0, mapping: str = "jordan_wigner"
) -> Hamiltonian:
    """Hubbard dimer on two sites (4 spin-orbitals: 0↑, 0↓, 1↑, 1↓).

    H = -t ∑_σ (c†_{0σ} c_{1σ} + h.c.) + U (n_{0↑} n_{0↓} + n_{1↑} n_{1↓})
    """
    a_fn, adag_fn = _ladder_operators(mapping)

    h_hop_up = (
        adag_fn(0, 4) @ a_fn(2, 4)
        + adag_fn(2, 4) @ a_fn(0, 4)
    )
    h_hop_dn = (
        adag_fn(1, 4) @ a_fn(3, 4)
        + adag_fn(3, 4) @ a_fn(1, 4)
    )
    h_hop = h_hop_up + h_hop_dn

    n0 = number_operator(0, 4, mapping=mapping)
    n1 = number_operator(1, 4, mapping=mapping)
    n2 = number_operator(2, 4, mapping=mapping)
    n3 = number_operator(3, 4, mapping=mapping)
    h_u = (n0 @ n1) + (n2 @ n3)

    h = (-float(t) * h_hop) + (float(U) * h_u)
    return h.simplify()


def occupation_block(
    hamiltonian: Hamiltonian | np.ndarray,
    n_particles: int = 2,
    n_orbitals: int | None = None,
) -> np.ndarray:
    """Restrict a Hamiltonian matrix to computational basis states with bit count `n_particles`."""
    h_mat = (
        hamiltonian.to_matrix()
        if isinstance(hamiltonian, Hamiltonian)
        else np.asarray(hamiltonian, dtype=complex)
    )
    dim = h_mat.shape[0]
    if n_orbitals is None:
        n_orbitals = int(np.round(np.log2(dim)))
    indices = [idx for idx in range(dim) if idx.bit_count() == n_particles]
    return h_mat[np.ix_(indices, indices)]


def hubbard_dimer_occupation_block(
    t: float = 1.0, U: float = 0.0, n_particles: int = 2
) -> np.ndarray:
    """Restrict hubbard_dimer(t, U) to computational basis states with bit count `n_particles`."""
    return occupation_block(hubbard_dimer(t=t, U=U), n_particles=n_particles, n_orbitals=4)


def tag_sectors(
    state: np.ndarray,
    n_spatial: int,
    tol: float = 1e-8,
    mapping: str = "jordan_wigner",
) -> tuple[int, float] | list[tuple[int, float]]:
    """Tag an eigenvector or collection of eigenvectors by ⟨N⟩ and ⟨S_z⟩.

    `state` is read in the qubit basis of `mapping`, Jordan-Wigner by default.
    Rejects any state whose ⟨N⟩ is not within `tol` (default 1e-8) of an integer.
    """
    arr = np.asarray(state, dtype=complex)
    n_orbitals = 2 * n_spatial
    ntot = total_number_operator(n_orbitals, mapping=mapping)
    sztot = total_spin_z(n_spatial, mapping=mapping)

    def _tag_single(vec: np.ndarray) -> tuple[int, float]:
        norm = np.linalg.norm(vec)
        if norm == 0:
            raise ValueError("state has zero norm")
        ket = vec / norm
        exp_n = ntot.expectation(ket)
        if abs(exp_n - round(exp_n)) > tol:
            raise ValueError(f"<N> = {exp_n} is not within {tol} of an integer")
        n_int = int(round(exp_n))
        exp_sz = sztot.expectation(ket)
        val_sz = float(np.round(exp_sz, 8))
        if abs(val_sz) < 1e-8:
            val_sz = 0.0
        return n_int, val_sz

    if arr.ndim == 1 or (arr.ndim == 2 and (arr.shape[0] == 1 or arr.shape[1] == 1)):
        return _tag_single(arr.reshape(-1))
    elif arr.ndim == 2:
        return [_tag_single(arr[:, col]) for col in range(arr.shape[1])]
    else:
        raise ValueError("state must be a 1D vector or 2D matrix of eigenvectors")


tag_sector = tag_sectors
sector = tag_sectors


def one_particle_rdm(
    state: np.ndarray, n: int, mapping: str = "jordan_wigner"
) -> np.ndarray:
    """One-particle reduced density matrix γ_{pq} = ⟨a†_p a_q⟩ for n orbitals.

    `state` is read in the qubit basis of `mapping`, Jordan-Wigner by default.
    """
    arr = np.asarray(state, dtype=complex)
    dim = 1 << n
    a_fn, adag_fn = _ladder_operators(mapping)
    a_mats = [a_fn(p, n).to_matrix() for p in range(n)]
    adag_mats = [adag_fn(p, n).to_matrix() for p in range(n)]

    gamma = np.zeros((n, n), dtype=complex)
    if arr.ndim == 1 or (arr.ndim == 2 and (arr.shape[0] == 1 or arr.shape[1] == 1)):
        ket = arr.reshape(-1)
        norm = np.linalg.norm(ket)
        if norm == 0:
            raise ValueError("state has zero norm")
        ket = ket / norm
        if ket.size != dim:
            raise ValueError(f"state size {ket.size} does not match 2^{n} = {dim}")
        for p in range(n):
            for q in range(n):
                op = adag_mats[p] @ a_mats[q]
                gamma[p, q] = np.vdot(ket, op @ ket)
    elif arr.ndim == 2 and arr.shape == (dim, dim):
        tr = np.trace(arr)
        if abs(tr) == 0:
            raise ValueError("density matrix has zero trace")
        rho = arr / tr
        for p in range(n):
            for q in range(n):
                op = adag_mats[p] @ a_mats[q]
                gamma[p, q] = np.trace(rho @ op)
    else:
        raise ValueError(f"state must be a ket (size {dim}) or density matrix ({dim}x{dim}) for {n} fermions")

    return gamma


def natural_occupations(
    state_or_rdm: np.ndarray, n: int | None = None, mapping: str = "jordan_wigner"
) -> np.ndarray:
    """Natural orbital occupation numbers (eigenvalues of one-particle RDM sorted descending)."""
    arr = np.asarray(state_or_rdm, dtype=complex)
    if n is not None and (arr.ndim == 1 or (arr.ndim == 2 and arr.shape[0] == (1 << n))):
        rdm = one_particle_rdm(arr, n, mapping=mapping)
    elif arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        if n is not None and arr.shape[0] != n:
            rdm = one_particle_rdm(arr, n, mapping=mapping)
        else:
            rdm = arr
    elif n is not None:
        rdm = one_particle_rdm(arr, n, mapping=mapping)
    else:
        raise ValueError("Must provide n or a square one-particle RDM matrix")
    evals = np.linalg.eigvalsh(rdm)
    return np.sort(np.real(evals))[::-1]
