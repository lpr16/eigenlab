"""Electronic and molecular Hamiltonians from one- and two-electron integrals."""

from __future__ import annotations

import numpy as np

from eigenlab.fermions import (
    bravyi_kitaev_a,
    bravyi_kitaev_adag,
    jordan_wigner_a,
    jordan_wigner_adag,
)
from eigenlab.hamiltonian import Hamiltonian, eigensystem, spectrum
from eigenlab.integrals import (
    ao_integrals,
    contracted_nuclear_attraction,
    h2_symmetry_orbitals,
    lowdin,
    mo_transform,
    rhf,
    sto3g_1s,
)


def integral_hamiltonian(
    one_electron: np.ndarray,
    two_electron: np.ndarray,
    nuclear_repulsion: float = 0.0,
    mapping: str = "jordan_wigner",
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
        - mapping is 'jordan_wigner' or 'bravyi_kitaev'
    """
    h1 = np.asarray(one_electron, dtype=complex)
    h2 = np.asarray(two_electron, dtype=complex)
    n = h1.shape[0]
    if h1.shape != (n, n):
        raise ValueError(f"one_electron must have shape ({n}, {n})")
    if h2.shape != (n, n, n, n):
        raise ValueError(f"two_electron must have shape ({n}, {n}, {n}, {n})")

    # Precompute fermionic operators according to selected mapping
    from collections import defaultdict

    map_norm = mapping.lower()
    if map_norm in ("jordan_wigner", "jw"):
        a_ops = [jordan_wigner_a(p, n) for p in range(n)]
        adag_ops = [jordan_wigner_adag(p, n) for p in range(n)]
    elif map_norm in ("bravyi_kitaev", "bk"):
        a_ops = [bravyi_kitaev_a(p, n) for p in range(n)]
        adag_ops = [bravyi_kitaev_adag(p, n) for p in range(n)]
    else:
        raise ValueError(f"Unknown mapping: {mapping}. Must be 'jordan_wigner' or 'bravyi_kitaev'")

    acc: dict[str, complex] = defaultdict(complex)
    if abs(nuclear_repulsion) > 1e-12:
        acc["I" * n] += complex(nuclear_repulsion)

    # One-electron terms: ∑_{p,q} h_{pq} a†_p a_q
    for p in range(n):
        for q in range(n):
            coeff = h1[p, q]
            if abs(coeff) > 1e-12:
                op = adag_ops[p] @ a_ops[q]
                for c_op, lbl in op.terms:
                    acc[lbl] += coeff * c_op

    # Precompute pairs a†_p a†_r (p != r) and a_s a_q (s != q)
    adag_pairs: dict[tuple[int, int], Hamiltonian] = {}
    a_pairs: dict[tuple[int, int], Hamiltonian] = {}
    for p in range(n):
        for r in range(n):
            if p != r:
                adag_pairs[(p, r)] = (adag_ops[p] @ adag_ops[r]).simplify()
                a_pairs[(p, r)] = (a_ops[p] @ a_ops[r]).simplify()

    # Two-electron terms: (1/2) ∑_{p,q,r,s} (pq|rs) a†_p a†_r a_s a_q
    for p in range(n):
        for r in range(n):
            if p == r:
                continue
            left_op = adag_pairs[(p, r)]
            for q in range(n):
                for s in range(n):
                    if s == q:
                        continue
                    coeff = h2[p, q, r, s]
                    if abs(coeff) > 1e-12:
                        op = left_op @ a_pairs[(s, q)]
                        half_coeff = 0.5 * coeff
                        for c_op, lbl in op.terms:
                            acc[lbl] += half_coeff * c_op

    filtered = [(val, lbl) for lbl, val in acc.items() if abs(val) > 1e-12]
    if not filtered:
        filtered = [(0.0j, "I" * n)]
    return Hamiltonian(filtered)


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


# Energy of an isolated hydrogen atom in STO-3G minimal basis (Hartree)
# Asymptote for two dissociated hydrogen atoms is 2 * H_ATOM_STO3G_ENERGY = -0.933164 Hartree
H_ATOM_STO3G_ENERGY = -0.466582


def h2_sto3g_integrals(
    bond_length_au: float = 1.401,
) -> tuple[np.ndarray, np.ndarray, float]:
    r"""Return STO-3G molecular integrals for H2 at any positive bond length in bohr (a.u.).

    Parameters
    ----------
    bond_length_au : float
        Internuclear distance R > 0 in atomic units (bohr).

    Returns
    -------
    h1 : np.ndarray
        Shape (2, 2) one-electron spatial orbital integrals in atomic units (Hartree),
        in the symmetry-adapted minimal MO basis [sigma_g, sigma_u].
    h2 : np.ndarray
        Shape (2, 2, 2, 2) two-electron spatial orbital integrals in chemist notation (pq|rs).
    nuclear_repulsion : float
        Nuclear repulsion energy V_nuc = 1/R in Hartree.
    """
    r_val = float(bond_length_au)
    if r_val <= 0.0:
        raise ValueError(f"bond_length_au must be positive, got {bond_length_au}")

    atoms = [
        (1.0, 1.24, [0.0, 0.0, 0.0]),
        (1.0, 1.24, [0.0, 0.0, r_val]),
    ]
    S, h, eri, v_nuc = ao_integrals(atoms)
    C = h2_symmetry_orbitals(S)
    h1, h2 = mo_transform(C, h, eri)
    return h1, h2, v_nuc


def h2_sto3g_hamiltonian(
    bond_length_au: float = 1.401,
    mapping: str = "jordan_wigner",
) -> Hamiltonian:
    """Build the second-quantized Pauli Hamiltonian for H2 in STO-3G basis.

    Returns the simplified 4-qubit Hamiltonian including nuclear repulsion.
    """
    h1, h2, v_nuc = h2_sto3g_integrals(bond_length_au)
    h1_spin, h2_spin = spatial_to_spin_orbital(h1, h2)
    return integral_hamiltonian(h1_spin, h2_spin, nuclear_repulsion=v_nuc, mapping=mapping)


def h2_energy_gradient(R: float, step: float = 1e-4) -> float:
    """Return the central-difference derivative dE/dR of the H2 STO-3G full-CI ground energy."""
    r_val = float(R)
    h_val = float(step)
    if h_val <= 0.0 or r_val - h_val <= 0.0:
        raise ValueError(f"Require R > step > 0, got R={R}, step={step}")
    e_plus = float(spectrum(h2_sto3g_hamiltonian(r_val + h_val))[0])
    e_minus = float(spectrum(h2_sto3g_hamiltonian(r_val - h_val))[0])
    return (e_plus - e_minus) / (2.0 * h_val)


def h2_electrostatic_hellmann_feynman(R: float = 1.401, step: float = 1e-4) -> float:
    """Electrostatic Hellmann-Feynman derivative d<V_ne + V_nn>/dR with the STO-3G basis frozen at R."""
    r_val = float(R)
    h_val = float(step)
    if h_val <= 0.0 or r_val - h_val <= 0.0:
        raise ValueError(f"Require R > step > 0, got R={R}, step={step}")

    # Basis functions and MO coefficients frozen at R
    r0 = np.array([0.0, 0.0, 0.0])
    r1 = np.array([0.0, 0.0, r_val])
    chi0 = sto3g_1s(1.24, r0)
    chi1 = sto3g_1s(1.24, r1)
    basis_fns = [chi0, chi1]

    ao = ao_integrals([(1.0, 1.24, r0), (1.0, 1.24, r1)])
    c_mat = h2_symmetry_orbitals(ao.S)
    _, evecs = eigensystem(h2_sto3g_hamiltonian(r_val))
    psi = evecs[:, 0]

    def v_ne_ao(r_nuc: float) -> np.ndarray:
        centers = [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, r_nuc])]
        v_mat = np.zeros((2, 2), dtype=float)
        for i in range(2):
            for j in range(i, 2):
                val = sum(
                    contracted_nuclear_attraction(basis_fns[i], basis_fns[j], 1.0, c_k)
                    for c_k in centers
                )
                v_mat[i, j] = v_mat[j, i] = val
        return v_mat

    dv_ne_ao = (v_ne_ao(r_val + h_val) - v_ne_ao(r_val - h_val)) / (2.0 * h_val)
    dv_ne_mo = c_mat.T @ dv_ne_ao @ c_mat
    dv_ne_spin, zero_h2 = spatial_to_spin_orbital(dv_ne_mo, np.zeros((2, 2, 2, 2)))
    dv_nn = (1.0 / (r_val + h_val) - 1.0 / (r_val - h_val)) / (2.0 * h_val)
    dh_frozen = integral_hamiltonian(dv_ne_spin, zero_h2, nuclear_repulsion=dv_nn)
    return float(dh_frozen.expectation(psi))


def _h2_rhf_energy(R: float) -> float:
    ao = ao_integrals([(1.0, 1.24, [0.0, 0.0, 0.0]), (1.0, 1.24, [0.0, 0.0, float(R)])])
    return rhf(ao, n_electrons=2).energy


def h2_rhf_energy_gradient(R: float, step: float = 1e-4) -> float:
    """Return the central-difference derivative dE_RHF/dR of the H2 STO-3G RHF energy."""
    r_val = float(R)
    h_val = float(step)
    if h_val <= 0.0 or r_val - h_val <= 0.0:
        raise ValueError(f"Require R > step > 0, got R={R}, step={step}")
    return (_h2_rhf_energy(r_val + h_val) - _h2_rhf_energy(r_val - h_val)) / (2.0 * h_val)


def h2_equilibrium_bond_length(
    method: str = "fci",
    bracket: tuple[float, float] = (1.0, 2.0),
    tol: float = 1e-6,
) -> float:
    """Bisect the H2 energy gradient on ``bracket`` until the interval is shorter than ``tol``."""
    low, high = float(bracket[0]), float(bracket[1])
    if low <= 0.0 or high <= low or tol <= 0.0:
        raise ValueError(f"Invalid bracket {bracket} or tolerance {tol}")

    kind = method.lower()
    if kind == "fci":
        grad_fn = h2_energy_gradient
    elif kind == "rhf":
        grad_fn = h2_rhf_energy_gradient
    else:
        raise ValueError(f"Unknown method: {method}. Must be 'fci' or 'rhf'")

    g_low = grad_fn(low)
    g_high = grad_fn(high)
    if not (g_low < 0.0 < g_high):
        raise ValueError(f"Gradient does not change sign from negative to positive on {bracket}")

    while (high - low) >= tol:
        mid = 0.5 * (low + high)
        if grad_fn(mid) < 0.0:
            low = mid
        else:
            high = mid

    return 0.5 * (low + high)


# CODATA 2018 constants (Tiesinga et al., Rev. Mod. Phys. 93, 025010 (2021)):
# - 1H atomic mass in electron masses m_e: m(1H) / m_e = 1.00782503223 u * 1822.888486209 m_e/u = 1837.15264958 m_e
# - Hartree energy in inverse centimeters: E_h / (h c) = 2 R_inf = 219474.63136320 cm^-1
CODATA_2018_H_MASS_AU = 1837.15264958
CODATA_2018_HARTREE_TO_CM1 = 219474.63136320


def h2_force_constant(R: float, step: float = 1e-2) -> float:
    """Central second difference d^2E/dR^2 of the H2 STO-3G full-CI ground energy at ``R``."""
    r_val = float(R)
    h_val = float(step)
    if h_val <= 0.0 or r_val - h_val <= 0.0:
        raise ValueError(f"Require R > step > 0, got R={R}, step={step}")
    e_plus = float(spectrum(h2_sto3g_hamiltonian(r_val + h_val))[0])
    e_zero = float(spectrum(h2_sto3g_hamiltonian(r_val))[0])
    e_minus = float(spectrum(h2_sto3g_hamiltonian(r_val - h_val))[0])
    return (e_plus - 2.0 * e_zero + e_minus) / (h_val * h_val)


def h2_harmonic_frequency(
    R_e: float | None = None,
    step: float = 1e-2,
    m_h_au: float = CODATA_2018_H_MASS_AU,
) -> tuple[float, float]:
    """Return ``(omega_au, nu_cm1)`` for H2 in STO-3G at ``R_e`` using CODATA 2018 constants."""
    if R_e is None:
        R_e = h2_equilibrium_bond_length()
    k = h2_force_constant(R_e, step=step)
    if k <= 0.0:
        raise ValueError(f"Force constant k must be positive, got {k}")
    mu = 0.5 * float(m_h_au)
    omega_au = float(np.sqrt(k / mu))
    nu_cm1 = omega_au * CODATA_2018_HARTREE_TO_CM1
    return omega_au, nu_cm1


def heh_sto3g_integrals(
    bond_length_au: float = 1.4632,
) -> tuple[np.ndarray, np.ndarray, float]:
    r"""Return STO-3G molecular integrals for HeH+ at any positive bond length in bohr (a.u.).

    Reference: Szabo & Ostlund, Modern Quantum Chemistry: Introduction to Advanced
    Electronic Structure Theory (1996), Section 3.5.3 (pp. 170-179) and Table 3.5.

    Geometry:
        - Internuclear distance: R > 0 a.u. (bohr), default R = 1.4632 a.u.
        - Nuclear charges: Z_He = 2 (zeta_He = 2.0925), Z_H = 1 (zeta_H = 1.24)
        - Nuclear repulsion energy: V_nuc = 2 / R Hartree

    Integrals:
        Computed from STO-3G 1s Gaussians via restricted Hartree-Fock (`rhf`).
        The sign of each MO column is fixed so its largest-magnitude coefficient is positive.
        At R = 1.4632 a.u., the transcribed atomic orbital integrals (zeta_He = 2.0925, zeta_H = 1.24) are:
            h_11 = -2.652744, h_12 = h_21 = -1.347205, h_22 = -1.731828
            S_12 = 0.450770
            (11|11) = 1.307152, (21|11) = 0.437279, (21|21) = 0.177267
            (22|11) = 0.605703, (22|21) = 0.311795, (22|22) = 0.774608
        Canonical RHF molecular orbital expansion coefficients (Szabo Table 3.5):
            C = [[ 0.80191665, -0.78226573],
                 [ 0.33680182,  1.06844491]]
        yielding orthonormal spatial molecular orbital integrals in chemist notation.

    Returns
    -------
    h1 : np.ndarray
        Shape (2, 2) one-electron spatial orbital integrals in MO basis (Hartree).
    h2 : np.ndarray
        Shape (2, 2, 2, 2) two-electron spatial orbital integrals in chemist notation (pq|rs).
    nuclear_repulsion : float
        Nuclear repulsion energy V_nuc = 2/R in Hartree.
    """
    r_val = float(bond_length_au)
    if r_val <= 0.0:
        raise ValueError(f"bond_length_au must be positive, got {bond_length_au}")

    ao = ao_integrals(
        [
            (2.0, 2.0925, [0.0, 0.0, 0.0]),
            (1.0, 1.24, [0.0, 0.0, r_val]),
        ]
    )
    _, _, c_mat, _ = rhf(ao, n_electrons=2)
    h1, h2 = mo_transform(c_mat, ao.h, ao.eri)
    return h1, h2, ao.v_nuc


def heh_sto3g_hamiltonian(
    bond_length_au: float = 1.4632,
    mapping: str = "jordan_wigner",
) -> Hamiltonian:
    """Build the second-quantized Pauli Hamiltonian for HeH+ in STO-3G basis.

    Returns the simplified 4-qubit Hamiltonian including nuclear repulsion.
    """
    h1, h2, v_nuc = heh_sto3g_integrals(bond_length_au)
    h1_spin, h2_spin = spatial_to_spin_orbital(h1, h2)
    return integral_hamiltonian(h1_spin, h2_spin, nuclear_repulsion=v_nuc, mapping=mapping)


def h4_sto3g_integrals(
    a: float,
    b: float,
    orbitals: str = "lowdin",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return STO-3G molecular integrals ``(h1, h2, v_nuc)`` for rectangular H4 at ``(±a/2, ±b/2, 0)``.

    Uses Löwdin symmetrically orthogonalized orbitals by default (or RHF orbitals when ``orbitals="rhf"``).
    """
    a_val = float(a)
    b_val = float(b)
    if a_val <= 0.0 or b_val <= 0.0:
        raise ValueError(f"Rectangle side lengths a and b must be positive, got a={a}, b={b}")

    atoms = [
        (1.0, 1.24, [-0.5 * a_val, -0.5 * b_val, 0.0]),
        (1.0, 1.24, [-0.5 * a_val, 0.5 * b_val, 0.0]),
        (1.0, 1.24, [0.5 * a_val, -0.5 * b_val, 0.0]),
        (1.0, 1.24, [0.5 * a_val, 0.5 * b_val, 0.0]),
    ]
    ao = ao_integrals(atoms)
    orb_kind = orbitals.lower()
    if orb_kind == "lowdin":
        c_mat = lowdin(ao.S)
    elif orb_kind == "rhf":
        c_mat = rhf(ao, n_electrons=4).coefficients
    else:
        raise ValueError(f"Unknown orbitals={orbitals!r}; expected 'lowdin' or 'rhf'")
    h1, h2 = mo_transform(c_mat, ao.h, ao.eri)
    return h1, h2, ao.v_nuc


def h4_sto3g_hamiltonian(
    a: float,
    b: float,
    mapping: str = "jordan_wigner",
    orbitals: str = "lowdin",
) -> Hamiltonian:
    """Build the 8-qubit second-quantized Pauli Hamiltonian for rectangular H4 in STO-3G."""
    h1, h2, v_nuc = h4_sto3g_integrals(a, b, orbitals=orbitals)
    h1_spin, h2_spin = spatial_to_spin_orbital(h1, h2)
    return integral_hamiltonian(h1_spin, h2_spin, nuclear_repulsion=v_nuc, mapping=mapping)



