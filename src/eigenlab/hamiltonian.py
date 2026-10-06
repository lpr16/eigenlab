"""Spin Hamiltonians and their exact spectra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from eigenlab.pauli import pauli


@dataclass(frozen=True)
class Hamiltonian:
    """Hamiltonian represented as a list of (coefficient, pauli_string) terms."""

    terms: list[tuple[complex, str]]

    def __init__(self, terms: Sequence[tuple[complex | float, str]]):
        object.__setattr__(self, "terms", [(complex(c), str(l)) for c, l in terms])

    @property
    def qubits(self) -> int:
        if not self.terms:
            return 0
        return len(self.terms[0][1])

    def to_matrix(self) -> np.ndarray:
        return to_matrix(self.terms)

    def expectation(self, state: np.ndarray) -> float:
        return term_expectation(self.terms, state)

    def __add__(self, other: Hamiltonian | Sequence[tuple[complex | float, str]]) -> Hamiltonian:
        other_terms = other.terms if isinstance(other, Hamiltonian) else list(other)
        return Hamiltonian(list(self.terms) + [(complex(c), l) for c, l in other_terms])

    def __sub__(self, other: Hamiltonian | Sequence[tuple[complex | float, str]]) -> Hamiltonian:
        other_terms = other.terms if isinstance(other, Hamiltonian) else list(other)
        return Hamiltonian(list(self.terms) + [(-complex(c), l) for c, l in other_terms])

    def __mul__(self, scalar: complex | float) -> Hamiltonian:
        return Hamiltonian([(c * complex(scalar), l) for c, l in self.terms])

    def __rmul__(self, scalar: complex | float) -> Hamiltonian:
        return Hamiltonian([(c * complex(scalar), l) for c, l in self.terms])

    def __matmul__(self, other: Hamiltonian) -> Hamiltonian:
        from collections import defaultdict
        from eigenlab.pauli import pauli_mult

        if not self.terms or not other.terms:
            return Hamiltonian([])
        combined: dict[str, complex] = defaultdict(complex)
        for c1, l1 in self.terms:
            for c2, l2 in other.terms:
                phase, label = pauli_mult(l1, l2)
                combined[label] += c1 * c2 * phase
        filtered = [(val, l) for l, val in combined.items() if abs(val) > 1e-12]
        return Hamiltonian(filtered if filtered else [(0.0, self.terms[0][1])])

    def dagger(self) -> Hamiltonian:
        return Hamiltonian([(complex(c).conjugate(), l) for c, l in self.terms])

    def simplify(self, tol: float = 1e-12) -> Hamiltonian:
        from collections import defaultdict

        combined: dict[str, complex] = defaultdict(complex)
        for c, l in self.terms:
            combined[l] += complex(c)
        filtered = [(val, l) for l, val in combined.items() if abs(val) > tol]
        if not filtered and self.terms:
            return Hamiltonian([(0.0, self.terms[0][1])])
        return Hamiltonian(filtered)


def to_matrix(hamiltonian: Hamiltonian | Sequence[tuple[complex | float, str]]) -> np.ndarray:
    """Build the matrix representation of a Hamiltonian for n <= 8 qubits."""
    terms = hamiltonian.terms if isinstance(hamiltonian, Hamiltonian) else list(hamiltonian)
    if not terms:
        raise ValueError("term list cannot be empty")
    qubits = len(terms[0][1])
    if qubits > 8:
        raise ValueError("matrices only supported for n <= 8 qubits")
    dim = 1 << qubits
    mat = np.zeros((dim, dim), dtype=complex)
    for coeff, label in terms:
        if len(label) != qubits:
            raise ValueError("all Pauli terms must have the same number of qubits")
        mat += coeff * pauli(label)
    return mat


def term_expectation(
    hamiltonian: Hamiltonian | Sequence[tuple[complex | float, str]],
    state: np.ndarray,
) -> float:
    """⟨ψ|H|ψ⟩ evaluated term by term without materializing the full Hamiltonian matrix."""
    terms = hamiltonian.terms if isinstance(hamiltonian, Hamiltonian) else list(hamiltonian)
    ket = np.asarray(state, dtype=complex).reshape(-1)
    norm = np.linalg.norm(ket)
    if norm == 0:
        raise ValueError("state has zero norm")
    ket = ket / norm
    total = 0.0 + 0.0j
    for coeff, label in terms:
        op = pauli(label)
        total += coeff * np.vdot(ket, op @ ket)
    if abs(total.imag) > 1e-8:
        raise ValueError("expectation of a non-Hermitian operator")
    return float(total.real)


def heisenberg_dimer_terms() -> Hamiltonian:
    """Term list for H = XX + YY + ZZ."""
    return xxz_dimer_terms(delta=1.0)


def heisenberg_dimer() -> np.ndarray:
    """H = X⊗X + Y⊗Y + Z⊗Z. Singlet at −3, triplet at +1."""
    return heisenberg_dimer_terms().to_matrix()


def xxz_dimer_terms(delta: float = 1.0) -> Hamiltonian:
    """Term list for XXZ dimer: H = XX + YY + Δ ZZ."""
    return Hamiltonian([(1.0, "XX"), (1.0, "YY"), (float(delta), "ZZ")])


def xxz_dimer(delta: float = 1.0) -> np.ndarray:
    """H = XX + YY + Δ ZZ."""
    return xxz_dimer_terms(delta=delta).to_matrix()


def transverse_ising_terms(
    qubits: int, coupling: float = 1.0, field: float = 1.0
) -> Hamiltonian:
    """Term list for open-chain transverse-field Ising, H = −J Σ Z_i Z_{i+1} − h Σ X_i."""
    if qubits < 2:
        raise ValueError("qubits must be at least 2")
    terms: list[tuple[complex | float, str]] = []
    for site in range(qubits - 1):
        label = ["I"] * qubits
        label[site] = "Z"
        label[site + 1] = "Z"
        terms.append((-coupling, "".join(label)))
    for site in range(qubits):
        label = ["I"] * qubits
        label[site] = "X"
        terms.append((-field, "".join(label)))
    return Hamiltonian(terms)


def transverse_ising(
    qubits: int, coupling: float = 1.0, field: float = 1.0
) -> np.ndarray:
    """Open-chain transverse-field Ising, H = −J Σ Z_i Z_{i+1} − h Σ X_i."""
    return transverse_ising_terms(qubits, coupling=coupling, field=field).to_matrix()


def eigensystem(hamiltonian: np.ndarray | Hamiltonian) -> tuple[np.ndarray, np.ndarray]:
    """Ascending energies and orthonormal eigenvectors (as columns) of a Hermitian operator."""
    if isinstance(hamiltonian, Hamiltonian):
        mat = hamiltonian.to_matrix()
    else:
        mat = np.asarray(hamiltonian, dtype=complex)
    values, vectors = np.linalg.eigh(mat)
    return np.real(values), vectors


diagonalize = eigensystem
eigen = eigensystem


def spectrum(hamiltonian: np.ndarray | Hamiltonian) -> np.ndarray:
    """Ascending eigenvalues of a Hermitian operator."""
    energies, _ = eigensystem(hamiltonian)
    return energies
