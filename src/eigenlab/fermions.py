"""Fermionic creation and annihilation operators via Jordan-Wigner transformation."""

from __future__ import annotations

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


def number_operator(p: int, n: int) -> Hamiltonian:
    """Number operator n_p = a†_p a_p = (I - Z_p) / 2."""
    if not 0 <= p < n:
        raise ValueError(f"orbital {p} out of range for {n} fermions")
    label_i = "I" * n
    label_z = "I" * p + "Z" + "I" * (n - 1 - p)
    return Hamiltonian([(0.5, label_i), (-0.5, label_z)])


def total_number_operator(n: int) -> Hamiltonian:
    """Total number operator N = ∑_{p=0}^{n-1} a†_p a_p."""
    terms: list[tuple[complex | float, str]] = [(0.5 * n, "I" * n)]
    for p in range(n):
        label_z = "I" * p + "Z" + "I" * (n - 1 - p)
        terms.append((-0.5, label_z))
    return Hamiltonian(terms)


def tight_binding_chain(n: int, t: float = 1.0) -> Hamiltonian:
    """Open tight-binding chain on n spinless fermions: H = -t ∑_i (a†_i a_{i+1} + h.c.)."""
    if n < 2:
        raise ValueError("tight binding chain requires at least 2 sites")
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
