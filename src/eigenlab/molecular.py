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


def h2_sto3g_integrals(
    bond_length_au: float = 1.401,
) -> tuple[np.ndarray, np.ndarray, float]:
    r"""Return STO-3G molecular integrals for H2 at specified bond length in bohr (a.u.).

    Parameters
    ----------
    bond_length_au : float
        Internuclear distance R in atomic units (bohr). Defaults to 1.401 a.u.
        (approx. 0.7414 Å), the equilibrium bond length transcribed from:
        - Whitfield et al., Mol. Phys. 109, 735-750 (2011), arXiv:1001.3855, Table 7
        - Seeley et al., J. Chem. Phys. 137, 224109 (2012), Table III
        - Szabo & Ostlund, Modern Quantum Chemistry (1996), Section 3.5 & Appendix D

    Returns
    -------
    h1 : np.ndarray
        Shape (2, 2) one-electron spatial orbital integrals in atomic units (Hartree).
    h2 : np.ndarray
        Shape (2, 2, 2, 2) two-electron spatial orbital integrals in chemist notation (pq|rs).
    nuclear_repulsion : float
        Nuclear repulsion energy V_nuc = 1/R in Hartree.
    """
    if np.isclose(bond_length_au, 1.401, atol=1e-3):
        # Equilibrium geometry: R = 1.401 a.u. (0.7414 Å)
        # Nuclear repulsion V_nuc = 1/R = 1/1.401 = 0.71377587... Hartree
        v_nuc = 1.0 / 1.401
        h1 = np.zeros((2, 2), dtype=float)
        h1[0, 0] = -1.252477
        h1[1, 1] = -0.475934

        h2 = np.zeros((2, 2, 2, 2), dtype=float)
        h2[0, 0, 0, 0] = 0.674493
        h2[1, 1, 1, 1] = 0.697397
        h2[0, 0, 1, 1] = 0.663472
        h2[1, 1, 0, 0] = 0.663472
        for p, q, r, s in [(0, 1, 1, 0), (1, 0, 0, 1), (0, 1, 0, 1), (1, 0, 1, 0)]:
            h2[p, q, r, s] = 0.181287
        return h1, h2, v_nuc
    else:
        raise ValueError(f"Unsupported bond length: {bond_length_au} a.u.")


def h2_sto3g_hamiltonian(
    bond_length_au: float = 1.401,
) -> Hamiltonian:
    """Build the second-quantized Pauli Hamiltonian for H2 in STO-3G basis.

    Returns the simplified 4-qubit Hamiltonian including nuclear repulsion.
    """
    h1, h2, v_nuc = h2_sto3g_integrals(bond_length_au)
    h1_spin, h2_spin = spatial_to_spin_orbital(h1, h2)
    return integral_hamiltonian(h1_spin, h2_spin, nuclear_repulsion=v_nuc)

