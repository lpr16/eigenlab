# Long haul

Eigenlab is exact linear algebra for at most eight qubits. This haul takes it from kets and two spin models to a Jordan–Wigner hydrogen molecule whose energy is checked by the same eigensolver. Twenty phases, in order. Each one is a single change set with a closed-form or cited number as the test.

Do not start phase k+1 until phase k is green under `pytest`.

## Contract

Stay inside this stack: Python 3.11, NumPy, pytest. Do not add Qiskit, PennyLane, OpenFermion, or PySCF.

`pauli("XZ")` already means qubit 0 is the leftmost character and the most significant bit of `basis(index, qubits)`. Every later mapping uses that order. Do not flip it.

Build matrices only for n ≤ 8. Once a Hamiltonian is a list of Pauli terms, keep it that way and materialize a matrix only to check the list.

A test number is either a few lines of algebra written next to the assertion, or a value copied from a citation named in the test module. Do not invent an H₂ energy to make a test pass.

Out of this haul: variational algorithms, shot noise, hardware, Bravyi–Kitaev, and any second molecule. Do not bump the package version.

## States

### 1. Bloch vector

Add `bloch` for a one-qubit ket.

- `|0⟩ → (0, 0, 1)`
- `|+⟩ → (1, 0, 0)`
- `|+i⟩ → (0, 1, 0)`

A product state of two qubits must be rejected. Pure states lie on the unit sphere.

### 2. Density operator

Add `density` and `purity = Tr(ρ²)`.

Every current ket has purity 1. The equal mixture of `|0⟩` and `|1⟩` has purity 1/2. `density` of an unnormalized ket matches the normalized one.

### 3. Partial trace

Add `partial_trace(rho, qubits, keep)`.

`|00⟩` traced over qubit 1 is `|0⟩⟨0|`. Either reduction of `bell("phi+")` is `I/2`. Tracing both sides returns `1`.

### 4. Two-qubit entanglement

Add the Schmidt spectrum across the middle cut, the entanglement entropy in bits, and the concurrence.

All four Bell states have concurrence 1 and entropy 1. `|00⟩`, `|+⟩⊗|0⟩`, and a phased product state have concurrence 0. Concurrence stays in `[0, 1]`.

## Operators and spectra

### 5. Pauli multiplication without a matrix

Add multiplication of two equal-length labels to a phase in `{1, -1, i, -i}` and a resulting label.

Check `XY = iZ`, `XZ = -iY`, `YZ = iX`, and that every length-2 pair matches `pauli(a) @ pauli(b)`.

### 6. Hamiltonians as term lists

Represent `H` as coefficients times labels. `to_matrix` and a term-list expectation must reproduce `heisenberg_dimer`, `transverse_ising`, and `expectation`.

Keep the existing functions as wrappers so the current tests stay.

### 7. Eigenvectors

Return ascending energies and orthonormal eigenvectors.

The ground state of `XX + YY + ZZ` is the singlet `(|01⟩ − |10⟩) / √2`, up to a global phase. The triplet is a subspace: the projector onto the three states at energy `+1` has rank 3. `spectrum` stays the eigenvalues alone.

### 8. XXZ dimer

Add `H = XX + YY + Δ ZZ`. From the triplet and singlet:

- `|00⟩` and `|11⟩` have energy `Δ`
- the symmetric Bell state has energy `2 − Δ`
- the singlet has energy `−2 − Δ`

`Δ = 1` matches phase 7. `Δ = 0` has energies `−2, 0, 0, 2`.

### 9. Transverse Ising, limits you can do by hand

For two sites and zero field, `H = −J ZZ` has energies `−J, −J, +J, +J`.

At large field the ground state is `|+⟩⊗n`, with `⟨∑ X⟩ → n` and `⟨∑ Z⟩ → 0`. Assert both observables on the ground eigenvector for `n = 3`, `J = 1`, `h = 0` and `h = 20`.

## Time

### 10. Exact evolution

`evolve(H, ψ, t)` from the spectral decomposition, `U = V exp(−i E t) V†`.

Norm and `⟨H⟩` are constant. `t = 0` and `H = 0` leave the state fixed. Evolution by `2π` under a Hamiltonian with integer spectrum returns the same ray.

### 11. One spin, closed form

Take `H = (ω/2) Z` and start in `|+⟩`. Then `⟨X(t)⟩ = cos(ω t)`.

Derive `⟨Y(t)⟩` in the test from the same Schrödinger equation and assert that sign. Check `t = 0`, `π/(2ω)`, and `π/ω`.

### 12. First-order Trotter step

For a term list, one step is `∏ exp(−i c_k P_k Δt)`, using `P² = I` so each factor is `cos(θ) I − i sin(θ) P`.

One term matches `evolve`. Two commuting terms (`ZI` and `IZ`, or `ZZ` and `II`) match `evolve` at any `Δt`.

### 13. Error when the terms do not commute

Use two-site transverse Ising, where `ZZ` does not commute with `X`. The Heisenberg dimer is the wrong example: `XX`, `YY`, and `ZZ` commute there.

Measure the Frobenius distance of one first-order step to the exact unitary. Halving `Δt` must cut that distance by about four.

Add a Strang splitting and show its distance falls by about eight. Fit neither exponent by loosening the assertion until it passes. Use several step sizes and a fixed tolerance around 2 and 3 on the log-log slope.

## Fermions

### 14. Jordan–Wigner

Map `a_p` and `a†_p` to Pauli strings with the frozen qubit order.

For `n ≤ 4`, the matrices must satisfy `{a_i, a†_j} = δ_ij I`, `{a_i, a_j} = 0`, and `a_i² = 0`. The vacuum is `basis(0, n)`. `a†_p |0⟩` is a single computational-basis vector.

### 15. Open tight-binding chain

`H = −t ∑_i (a†_i a_{i+1} + h.c.)` on spinless fermions.

Eigenvalues are `−2t cos(π k / (N+1))` for `k = 1…N`. The number operator `∑ a† a` commutes with `H`, and its expectation on every eigenvector is an integer.

### 16. Hubbard dimer

Two sites, spin up and spin down, hopping `t`, on-site `U`.

At `t = 0` the energies are `0` and `U` with degeneracies counted by occupation. At half filling, reduce by hand to the `N = 2`, `S_z = 0` singlet subspace, write the `3×3`, and match its lowest eigenvalue. That root is `(U − √(U² + 16 t²)) / 2`.

Check `U = 0` (energy `−2|t|`) and `t = 0` (energy `0`).

### 17. Sectors

Tag eigenvectors by `⟨N⟩` and `⟨S_z⟩`.

The Hubbard ground state from phase 16 sits at `N = 2`, `S_z = 0`. A one-electron hopping eigenstate sits at `N = 1`. Reject a vector whose `⟨N⟩` is not within `1e-8` of an integer.

## The molecule

### 18. Integral Hamiltonian

Build a Pauli term list from one- and two-electron arrays in chemist notation, `(pq|rs)`, stated in the docstring.

Test it on a two-orbital diagonal toy: the expectation value on a computational occupation state equals `h_ii + h_jj + (ii|jj) − (ij|ji)`, computed from the integrals directly and from the term list.

### 19. H₂, one geometry

Transcribe STO-3G integrals for one published bond length from a named reference. Include nuclear repulsion the way that reference does. Diagonalize with the existing eigensolver.

The ground-state energy must match the cited full configuration-interaction number to the precision the citation actually prints. Record geometry, basis, and whether repulsion is included. If two references disagree, stop on this phase and write the disagreement into the test docstring.

### 20. Two more geometries, then the scope line

Add a shorter bond and a stretched bond from the same citation family, same basis, same integral convention. Assert those cited totals.

The stretched geometry is the check that the energy moves toward the sum of two STO-3G hydrogen atoms given by that source.

Then update `docs/scope.md` and the README table so Jordan–Wigner, the Trotter comparison, and this H₂ check are described as present. Name Bravyi–Kitaev as the thing this haul did not do.

## Done

After phase 20 the lab should still be what `docs/scope.md` already asks for: a state, a Hamiltonian, and a spectrum you can recompute by hand, now including one molecule whose number came from a book rather than from a fit.
