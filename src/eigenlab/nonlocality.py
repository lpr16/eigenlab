"""Bell operators whose quantum value beats every local assignment."""

from __future__ import annotations

import itertools

from eigenlab.hamiltonian import Hamiltonian


def mermin_operator(n: int) -> Hamiltonian:
    """Real part of ∏_k (X_k + i Y_k).

    The expansion keeps every X/Y string with an even number of Ys, and the
    sign of that string is (−1)^{#Y / 2}.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    terms: list[tuple[float, str]] = []
    for letters in itertools.product("XY", repeat=n):
        n_y = letters.count("Y")
        if n_y % 2:
            continue
        sign = float((-1) ** (n_y // 2))
        terms.append((sign, "".join(letters)))
    return Hamiltonian(terms).simplify()


def mermin_three() -> Hamiltonian:
    """Three-qubit Mermin operator ``XXX − XYY − YXY − YYX``."""
    return Hamiltonian(
        [(1.0, "XXX"), (-1.0, "XYY"), (-1.0, "YXY"), (-1.0, "YYX")]
    ).simplify()
