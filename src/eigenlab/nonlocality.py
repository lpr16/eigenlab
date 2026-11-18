"""Bell operators whose quantum value beats every local assignment."""

from __future__ import annotations

from eigenlab.hamiltonian import Hamiltonian


def mermin_three() -> Hamiltonian:
    """Three-qubit Mermin operator ``XXX − XYY − YXY − YYX``."""
    return Hamiltonian(
        [(1.0, "XXX"), (-1.0, "XYY"), (-1.0, "YXY"), (-1.0, "YYX")]
    ).simplify()
