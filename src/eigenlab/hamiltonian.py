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


def total_spin(qubits: int) -> Hamiltonian:
    """Total spin squared operator S² = S_- S_+ + S_z² + S_z for qubits.

    S_z^{(i)} = Z_i / 2 and S_+^{(i)} = (X_i + i Y_i) / 2.
    """
    if qubits < 1:
        raise ValueError("qubits must be at least 1")
    terms_sz: list[tuple[complex | float, str]] = []
    terms_sp: list[tuple[complex | float, str]] = []
    for i in range(qubits):
        lbl_z = ["I"] * qubits
        lbl_z[i] = "Z"
        terms_sz.append((0.5, "".join(lbl_z)))

        lbl_x = ["I"] * qubits
        lbl_x[i] = "X"
        terms_sp.append((0.5, "".join(lbl_x)))

        lbl_y = ["I"] * qubits
        lbl_y[i] = "Y"
        terms_sp.append((0.5j, "".join(lbl_y)))

    sz = Hamiltonian(terms_sz)
    sp = Hamiltonian(terms_sp)
    sm = sp.dagger()
    s2 = (sm @ sp) + (sz @ sz) + sz
    return s2.simplify()


total_spin_squared = total_spin


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


def transverse_ising_bdg_matrix(
    qubits: int, coupling: float = 1.0, field: float = 1.0
) -> np.ndarray:
    """Bogoliubov–de Gennes 2n x 2n matrix for the open-chain transverse-field Ising model."""
    if qubits < 1:
        raise ValueError("qubits must be at least 1")
    n = qubits
    A = np.zeros((n, n), dtype=float)
    B = np.zeros((n, n), dtype=float)
    for i in range(n):
        A[i, i] = 2.0 * float(field)
    for i in range(n - 1):
        A[i, i + 1] = -float(coupling)
        A[i + 1, i] = -float(coupling)
        B[i, i + 1] = -float(coupling)
        B[i + 1, i] = float(coupling)
    return np.block([[A, B], [-B, -A]])


def transverse_ising_bogoliubov_spectrum(
    qubits: int, coupling: float = 1.0, field: float = 1.0
) -> np.ndarray:
    """Exact spectrum of open transverse Ising chain computed via Bogoliubov free-fermion solution."""
    import itertools

    n = qubits
    bdg = transverse_ising_bdg_matrix(n, coupling=coupling, field=field)
    evals = np.sort(np.real(np.linalg.eigvals(bdg)))
    eps = evals[n:]
    energies = [
        sum((nu_k - 0.5) * eps[k] for k, nu_k in enumerate(nu))
        for nu in itertools.product([0, 1], repeat=n)
    ]
    return np.sort(np.array(energies, dtype=float))


def eigensystem(
    hamiltonian: np.ndarray | Hamiltonian,
    commuting: Sequence[np.ndarray | Hamiltonian] | np.ndarray | Hamiltonian | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Ascending energies and orthonormal eigenvectors (as columns) of a Hermitian operator."""
    if commuting is not None:
        return simultaneous_eigensystem(hamiltonian, commuting)
    if isinstance(hamiltonian, Hamiltonian):
        mat = hamiltonian.to_matrix()
    else:
        mat = np.asarray(hamiltonian, dtype=complex)
    values, vectors = np.linalg.eigh(mat)
    return np.real(values), vectors


def simultaneous_eigensystem(
    hamiltonian: np.ndarray | Hamiltonian,
    commuting: Sequence[np.ndarray | Hamiltonian] | np.ndarray | Hamiltonian,
) -> tuple[np.ndarray, np.ndarray]:
    """Simultaneously diagonalize a Hamiltonian and one or more commuting symmetries."""
    h_mat = (
        hamiltonian.to_matrix()
        if isinstance(hamiltonian, Hamiltonian)
        else np.asarray(hamiltonian, dtype=complex)
    )
    if isinstance(commuting, (Hamiltonian, np.ndarray)):
        comm_list = [commuting]
    else:
        comm_list = list(commuting)

    h_pert = h_mat.copy()
    primes = [np.sqrt(2.0), np.sqrt(3.0), np.sqrt(5.0), np.sqrt(7.0), np.sqrt(11.0)]
    for idx, c in enumerate(comm_list):
        c_mat = c.to_matrix() if isinstance(c, Hamiltonian) else np.asarray(c, dtype=complex)
        alpha = 1e-6 * primes[idx % len(primes)]
        h_pert += alpha * c_mat

    _, v = np.linalg.eigh(h_pert)
    energies = np.real(np.diag(v.conj().T @ h_mat @ v))
    order = np.argsort(energies)
    return energies[order], v[:, order]


diagonalize = eigensystem
eigen = eigensystem


def spectrum(hamiltonian: np.ndarray | Hamiltonian) -> np.ndarray:
    """Ascending eigenvalues of a Hermitian operator."""
    energies, _ = eigensystem(hamiltonian)
    return energies
