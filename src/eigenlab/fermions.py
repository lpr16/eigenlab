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


def total_number_operator(n: int) -> Hamiltonian:
    """Total number operator N = ∑_{p=0}^{n-1} a†_p a_p."""
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


def total_spin_z(n_spatial: int) -> Hamiltonian:
    """Total spin projection S_z = (1/2) ∑_i (n_{i↑} - n_{i↓}).

    Spin-orbitals are interleaved as: orbital 2*i is spin-up, 2*i + 1 is spin-down.
    """
    n_orbitals = 2 * n_spatial
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
    map_norm = mapping.lower()
    if map_norm in ("jordan_wigner", "jw"):
        a_fn = jordan_wigner_a
        adag_fn = jordan_wigner_adag
    elif map_norm in ("bravyi_kitaev", "bk"):
        a_fn = bravyi_kitaev_a
        adag_fn = bravyi_kitaev_adag
    else:
        raise ValueError(f"Unknown mapping: {mapping}. Must be 'jordan_wigner' or 'bravyi_kitaev'")

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


def tag_sectors(
    state: np.ndarray,
    n_spatial: int,
    tol: float = 1e-8,
) -> tuple[int, float] | list[tuple[int, float]]:
    """Tag an eigenvector or collection of eigenvectors by ⟨N⟩ and ⟨S_z⟩.

    Rejects any state whose ⟨N⟩ is not within `tol` (default 1e-8) of an integer.
    """
    arr = np.asarray(state, dtype=complex)
    n_orbitals = 2 * n_spatial
    ntot = total_number_operator(n_orbitals)
    sztot = total_spin_z(n_spatial)

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
