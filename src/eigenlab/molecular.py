"""Electronic and molecular Hamiltonians from one- and two-electron integrals."""

from __future__ import annotations

import numpy as np

from eigenlab.fermions import jordan_wigner_a, jordan_wigner_adag
from eigenlab.hamiltonian import Hamiltonian


def integral_hamiltonian(
    one_electron: np.ndarray,
    two_electron: np.ndarray,
    nuclear_repulsion: float = 0.0,
) -> Hamiltonian:
    r"""Build a Pauli term list from 1- and 2-electron integrals in chemist notation.

    The second-quantized electronic Hamiltonian is:
        H = E_nuc * I
          + \sum_{p, q} h_{pq} a_p^\dagger a_q
          + \frac{1}{2} \sum_{p, q, r, s} (pq|rs) a_p^\dagger a_r^\dagger a_s a_q

    where chemist notation (pq|rs) is:
        (pq|rs) = \int \phi_p^*(r_1) \phi_q(r_1) \frac{1}{|r_1 - r_2|} \phi_r^*(r_2) \phi_s(r_2) dr_1 dr_2

    Here:
        - one_electron is an (n, n) array of one-electron integrals h_{pq}
        - two_electron is an (n, n, n, n) array of two-electron integrals (pq|rs)
        - nuclear_repulsion is the scalar nuclear repulsion energy E_nuc
        - n is the number of spin-orbitals (and qubits)
    """
    h1 = np.asarray(one_electron, dtype=complex)
    h2 = np.asarray(two_electron, dtype=complex)
    n = h1.shape[0]
    if h1.shape != (n, n):
        raise ValueError(f"one_electron must have shape ({n}, {n})")
    if h2.shape != (n, n, n, n):
        raise ValueError(f"two_electron must have shape ({n}, {n}, {n}, {n})")

    terms: list[tuple[complex, str]] = []
    if abs(nuclear_repulsion) > 1e-12:
        terms.append((complex(nuclear_repulsion), "I" * n))

    H = Hamiltonian(terms)

    # Precompute Jordan-Wigner operators
    a_ops = [jordan_wigner_a(p, n) for p in range(n)]
    adag_ops = [jordan_wigner_adag(p, n) for p in range(n)]

    # One-electron terms: ∑_{p,q} h_{pq} a†_p a_q
    for p in range(n):
        for q in range(n):
            coeff = h1[p, q]
            if abs(coeff) > 1e-12:
                op = adag_ops[p] @ a_ops[q]
                H = H + (coeff * op)

    # Two-electron terms: (1/2) ∑_{p,q,r,s} (pq|rs) a†_p a†_r a_s a_q
    for p in range(n):
        for q in range(n):
            for r in range(n):
                for s in range(n):
                    coeff = h2[p, q, r, s]
                    if abs(coeff) > 1e-12:
                        op = adag_ops[p] @ adag_ops[r] @ a_ops[s] @ a_ops[q]
                        H = H + (0.5 * coeff * op)

    return H.simplify()


molecular_hamiltonian = integral_hamiltonian


def spatial_to_spin_orbital(
    h1_spatial: np.ndarray,
    h2_spatial: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert spatial-orbital integrals to spin-orbital integrals.

    For k spatial orbitals, produces 2*k spin-orbitals with interleaved spins:
    spin-orbital 2*i is spin-up, 2*i + 1 is spin-down.

    Chemist notation:
    (p q | r s)_{spin} = (mu nu | lam kap)_{spatial} * delta_{sp, sq} * delta_{sr, ss}
    h_{p q}^{spin} = h_{mu nu}^{spatial} * delta_{sp, sq}
    """
    k = h1_spatial.shape[0]
    n = 2 * k
    h1_spin = np.zeros((n, n), dtype=complex)
    h2_spin = np.zeros((n, n, n, n), dtype=complex)

    for p in range(n):
        mu, sp = divmod(p, 2)
        for q in range(n):
            nu, sq = divmod(q, 2)
            if sp == sq:
                h1_spin[p, q] = h1_spatial[mu, nu]

    for p in range(n):
        mu, sp = divmod(p, 2)
        for q in range(n):
            nu, sq = divmod(q, 2)
            if sp != sq:
                continue
            for r in range(n):
                lam, sr = divmod(r, 2)
                for s in range(n):
                    kap, ss = divmod(s, 2)
                    if sr == ss:
                        h2_spin[p, q, r, s] = h2_spatial[mu, nu, lam, kap]

    return h1_spin, h2_spin
