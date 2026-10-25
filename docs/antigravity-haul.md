# Antigravity haul

The first two hauls are in the tree. This one takes the lab from a spectrum to the questions a spectrum can answer: a symmetry block, a free-fermion mode, a circuit with an exact output, a channel, a perturbation. Fifty phases, in order. Each one is a single change set with a closed-form check.

Do not start phase k+1 until phase k is green under `pytest`.

This file is the spec for an agent, including Google Antigravity. A workflow file there is capped at 12,000 characters, so the spec stays here. Point `/goal` at one phase. If you are asked to do the whole haul, finish phase k, leave `pytest` green, then start phase k+1. Do not implement two phases behind one test run. Do not bump the package version. Completion of a phase is the new assertions passing, not a summary of intent.

## Contract

Stay inside this stack: Python 3.11, NumPy, pytest. Do not add Qiskit, PennyLane, OpenFermion, PySCF, or SciPy.

`pauli("XZ")` means qubit 0 is the leftmost character and the most significant bit of `basis(index, qubits)`. Orbital 0 is that same qubit. `spatial_to_spin_orbital` interleaves spins as `(↑, ↓, ↑, ↓, …)`. Jordan–Wigner stays the default. Do not flip any of these.

Build matrices only for n ≤ 8. A Pauli expectation may act on a larger ket by permuting amplitudes. Do not diagonalize anything above eight qubits.

A test number is a few lines of algebra written next to the assertion. Do not invent a decimal to make a test pass. Do not loosen a tolerance until a fit succeeds.

Out of this haul: variational algorithms, shot noise, hardware, a third molecule, a derivative of the H₂ curve, and exact diagonalization above eight qubits.

## Pauli strings as operators

### 1. Commutator and anticommutator

Add `pauli_comm` and `pauli_anti` for two equal-length labels. Each returns `(phase, label)` using `pauli_mult`.

- `[X, Z] = XZ − ZX = −2i Y`, and `{X, Z} = 0`.
- `[XX, YY] = 0` on two qubits. These two labels commute.
- The term-list commutator `[H₁, H₂] = H₁ H₂ − H₂ H₁`, using the existing `Hamiltonian.__matmul__`, matches the matrix commutator of `XI` and `ZI` to `1e-12`.

### 2. A Pauli as a signed permutation

Add `apply_pauli(label, ket)`. It flips bits and multiplies phases. It does not call `pauli` or allocate a `2^n × 2^n` matrix.

On one qubit, `Z|0⟩ = |0⟩`, `Z|1⟩ = −|1⟩`, `X` swaps the two amplitudes, `Y|0⟩ = i|1⟩`, `Y|1⟩ = −i|0⟩`. Qubit 0 is the most significant bit.

For every label of length 2 and 3, and every computational basis ket, `apply_pauli` matches `pauli(label) @ ket` to `1e-12`.

`term_expectation` and `expectation` of a label use this path. The test patches `eigenlab.pauli.pauli` and `eigenlab.hamiltonian.pauli` so they raise, then checks that `expectation` of `Z` on `basis(0, 1)` still returns `+1`.

For nine qubits, `apply_pauli("Z" + "I"*8, basis(0, 9))` equals that basis ket, and `to_matrix` of the same one-term Hamiltonian still raises `ValueError`. That expectation is not exact diagonalization.

### 3. Conjugation by a Pauli

Add `conjugate_pauli(target, conjugator)` for equal-length labels. Since `P² = I` up to the phase the product table already carries, this is conjugation of one label by another.

`XZX = −Z` and `YXY = −X` on one qubit. For every pair of length-2 labels, the conjugated label matches `P @ Q @ P` from the matrices, including the phase, to `1e-12`.

### 4. Hadamard and S

Add `conjugate_clifford(label, gate, qubit)` for `gate` in `{"H", "S"}`. It returns `(phase, label)` for `U P U†`, touching only the named qubit.

On that qubit, `H` sends `X → Z`, `Z → X`, `Y → −Y`. `S = diag(1, i)` sends `X → Y`, `Y → −X`, `Z → Z`.

For every length-2 label, both qubits, and both gates, the result matches the matrix conjugation to `1e-12`.

### 5. Stabilizer expectations

Add `expectation_label(state, label)` using `apply_pauli`.

- `bell("phi+")` has `⟨XX⟩ = ⟨ZZ⟩ = +1` and `⟨YY⟩ = −1`.
- `bell("psi-")` has `⟨XX⟩ = ⟨YY⟩ = ⟨ZZ⟩ = −1`.
- `basis(0, 2)` has `⟨ZI⟩ = ⟨IZ⟩ = +1` and `⟨XI⟩ = 0`.

## States

### 6. GHZ

Add `ghz(n)` for `2 ≤ n ≤ 8`, the ket `(|0…0⟩ + |1…1⟩) / √2`. For `n = 2` this is `bell("phi+")`.

For `n = 2, 3, 4`:

- The entanglement entropy across the middle cut is 1 bit.
- The one-qubit reduction has purity `1/2`.
- `⟨X⊗n⟩ = 1`, and `⟨Z_i Z_{i+1}⟩ = 1` for each neighboring pair.

### 7. W state

Add `w_state()` for three qubits,

```text
(|001⟩ + |010⟩ + |100⟩) / √3
```

in the existing order, so those kets are `basis(1, 3)`, `basis(2, 3)`, and `basis(4, 3)`.

The middle-cut entropy equals the binary entropy of `1/3`,

```text
-(1/3) log2(1/3) - (2/3) log2(2/3)
```

written that way in the test. Every two-qubit reduction has concurrence `2/3`.

### 8. Purification

Add `purify(rho)` for a one-qubit density matrix. From `ρ = Σ_i p_i |u_i⟩⟨u_i|`, return `Σ_i √p_i |u_i⟩ ⊗ |i⟩` on two qubits. The ancilla is qubit 1.

Tracing out qubit 1 returns `ρ`. `purify` of `I/2` has concurrence 1. The squared Schmidt spectrum equals the eigenvalues of `ρ`.

### 9. Negativity

Add `negativity(state)` for a two-qubit ket or density matrix. Reshape `ρ` as `(2, 2, 2, 2)` with axes `(row₀, row₁, col₀, col₁)` and transpose the qubit-1 axes. Negativity is the sum of the absolute values of the negative eigenvalues of that partial transpose.

All four Bell states have negativity `1/2`. `basis(0, 2)` and `|+⟩⊗|0⟩` have negativity 0.

The Werner state `p |Ψ⁻⟩⟨Ψ⁻| + (1−p) I/4` has negativity `max(0, (3p−1)/4)`. Check `p = 0`, `p = 1/3`, `p = 1/2`, and `p = 1`. At `p = 1/3` this is the separable boundary, so the value is 0, not a small positive number.

### 10. Entanglement of formation

Add `entanglement_of_formation(state)` in bits, from the concurrence `C`:

```text
h( (1 + √(1 − C²)) / 2 )
```

where `h` is the binary entropy and `h(0) = h(1) = 0` by continuity.

A Bell state gives 1. A product state gives 0. The Werner state at `p = 1` gives 1, and at `p = 1/3` gives 0.

## Measurement

### 11. Pauli probabilities

Add `pauli_probabilities(state, label)`. Return the probabilities of the `+1` and `−1` outcomes, in that order.

- `X` on `|+⟩` is `(1, 0)`.
- `Z` on `|+⟩` is `(1/2, 1/2)`.
- `ZZ` on `bell("phi+")` is `(1, 0)`.
- `Z` on `basis(1, 1)` is `(0, 1)`.

The two probabilities sum to 1.

### 12. Post-measurement state

Add `post_measure(state, label, outcome)` with `outcome` in `{+1, −1}`. Apply the projector `(I ± P) / 2` by the signed permutation, then normalize. Multiply by a phase so the first nonzero amplitude is real and positive.

`Z = +1` on `|+⟩` returns `basis(0, 1)`. `Z = −1` on `|+⟩` returns `basis(1, 1)`. An outcome with probability 0 raises `ValueError`.

### 13. Shannon entropy

Add `shannon_entropy(state)` in bits for a computational-basis measurement. Take `0 log 0 = 0`.

`basis(0, 1)` gives 0. `bell("phi+")` gives 1. `w_state()` gives `log2(3)`.

### 14. Helstrom

Add `helstrom(psi, phi)` for two pure kets and equal priors. The success probability is

```text
(1 + √(1 − |⟨ψ|φ⟩|²)) / 2
```

Identical states give 1. Orthogonal states give 1. `|0⟩` against `|+⟩` gives `(1 + √(1/2)) / 2`.

### 15. Swap test

Add `swap_test(psi, phi)` for two one-qubit kets. Build the three-qubit circuit. Do not return the overlap formula without applying the gates.

Ancilla is qubit 0, started in `|0⟩`. Data qubits are 1 and 2. Apply `H` on the ancilla, then the Fredkin gate that swaps qubits 1 and 2 when the ancilla is 1, then `H` on the ancilla again. Return the probability that the ancilla is 0.

That probability equals `(1 + |⟨ψ|φ⟩|²) / 2`. The test checks both the circuit output and the formula. `|0⟩` with `|0⟩` gives 1. `|0⟩` with `|1⟩` gives `1/2`.

## Circuits with an exact output

### 16. One Grover iteration

Add `grover_once(marked, qubits=2)`. Two qubits, one marked computational-basis index.

Start from the uniform ket. The oracle is diagonal, with `−1` in the marked entry. The diffuser is `2|s⟩⟨s| − I`. One application returns the marked basis ket up to a global phase: probability 1 on that index, for every `marked` in `{0, 1, 2, 3}`.

### 17. Quantum Fourier transform

Add `qft_matrix(n)` for `n = 2` and `n = 3`. Index rows and columns from 0. Entry `(j, k)` is `ω^{jk} / √N` with `N = 2^n` and `ω = exp(2πi / N)`.

The matrix is unitary to `1e-12`. `qft_matrix(n) @ basis(0, n)` is the uniform ket.

### 18. The clock shift

Add `cyclic_shift(n)`, the permutation with `T |k⟩ = |k+1 mod 2^n⟩`.

For `n = 2`, `Q† T Q` is diagonal, where `Q` is `qft_matrix(2)`. The diagonal entries, in column order `k = 0, 1, 2, 3`, are `ω^{−k}` for `ω = exp(2πi / 4)`, that is `1, −i, −1, i`.

### 19. CHSH

Add `chsh_operator()`. Qubit 0 is measured with `A = Z` and `A' = X`. Qubit 1 is measured with `B = (Z + X) / √2` and `B' = (Z − X) / √2`. The operator is

```text
A⊗B + A⊗B' + A'⊗B − A'⊗B'
```

`bell("phi+")` gives `2√2`. `basis(0, 2)` gives `√2`. Write the four correlators for `|00⟩` next to that assertion: `⟨Z⟩ = 1`, `⟨X⟩ = 0`, so the two `A'` terms vanish and the two `A` terms contribute `1/√2` each.

### 20. Relative entropy

Add `relative_entropy(rho, sigma)` in bits,

```text
Tr ρ log2 ρ − Tr ρ log2 σ
```

with `0 log 0 = 0`. `sigma` must be positive definite; otherwise raise `ValueError`.

`S(|0⟩⟨0| ‖ I/2) = 1`. `S(I/2 ‖ I/2) = 0`.

## Blocks, not a perturbation

### 21. Parity of the transverse Ising chain

Add `parity(n)`, the term list for `P = X⊗n`.

`P` commutes with `transverse_ising(n, J, h)` for `n = 2, 3, 4` and `(J, h) = (1, 1)`. The ground state at `n = 3`, `J = 1`, `h = 1` has `⟨P⟩ = +1`.

The even cat `(|0…0⟩ + |1…1⟩) / √2` has `⟨P⟩ = +1`. The odd cat `(|0…0⟩ − |1…1⟩) / √2` has `⟨P⟩ = −1`. Do not claim that `eigh` picks the even cat when `h = 0`; that ground space is degenerate.

### 22. Parity blocks

Add `parity_blocks(hamiltonian, n)`. Build the projectors `(I ± P) / 2`, restrict the matrix to each support, and diagonalize the two blocks. Return the two spectra.

Do not call `simultaneous_eigensystem`. For `n = 2, 3, 4` and `(J, h)` in `{(1, 0), (1, 0.5), (1, 1), (0.7, 1.3)}`, the sorted union matches `spectrum(transverse_ising(n, J, h))` to `1e-8`, degeneracies included.

### 23. Occupation block of the Hubbard dimer

Add a restriction of `hubbard_dimer` to the computational basis states whose bit count is 2. Diagonalize that block directly.

At `t = 1` and `U` in `{0, 4}`, the lowest eigenvalue is `(U − √(U² + 16 t²)) / 2`. The block has dimension `C(4, 2) = 6`. Do not call `simultaneous_eigensystem`.

### 24. Spin blocks of the Heisenberg dimer

`S_z = (ZI + IZ) / 2`. Restrict `heisenberg_dimer` to the `S_z = 0` basis states `|01⟩` and `|10⟩`, and to the one-dimensional spaces `|00⟩` and `|11⟩`.

The `S_z = 0` block has energies `−3` and `+1`. Each of `|00⟩` and `|11⟩` has energy `+1`.

### 25. The perturbed solver stays, and is labeled

Do not remove `simultaneous_eigensystem`. State in its docstring that it adds a small irrational multiple of each commuting operator and calls `eigh`. It is not a projector.

On the Hubbard dimer at `t = 1`, `U = 4`, its eigenvalues match the occupation-block spectrum from phase 23 to `1e-8`.

## Quadratic fermions

### 26. Su–Schrieffer–Heeger chain

Add `ssh_chain(n, v, w, mapping="jordan_wigner")` for even `n ≥ 2`. Open chain. Bond `i` has hopping `v` when `i` is even and `w` when `i` is odd, with the same fermion operators and the same minus sign as `tight_binding_chain`. When `v = w = t` the term list matches `tight_binding_chain(n, t)` to `1e-12` after `simplify`, for `n = 2` and `n = 4`.

The test builds the single-particle matrix itself: `M_{i,i+1} = M_{i+1,i} = −v` or `−w`. Many-body energies are all fillings of its eigenvalues.

For `n = 4` those eigenvalues are the four sign combinations

```text
[ ±w ± √(w² + 4v²) ] / 2
```

Check `(v, w) = (1, 0)`, `(0, 1)`, and `(1, 1)`. At `(0, 1)` the single-particle energies are `−1, 0, 0, +1`, and the two zero eigenvectors are the end sites.

### 27. Kitaev chain, two sites by hand

Add `kitaev_chain(n, t, delta, mu, mapping="jordan_wigner")`,

```text
H = Σ_i −μ (n_i − 1/2)
  + Σ_i [ −t (a†_i a_{i+1} + h.c.) + Δ (a_i a_{i+1} + a†_{i+1} a†_i) ]
```

using the Jordan–Wigner operators in that product order. `n_i = a†_i a_i`.

At `Δ = 0` and `μ = 0` this matches `tight_binding_chain`.

For `n = 2`, write these blocks in the test. Order `(|00⟩, |11⟩)` and `(|10⟩, |01⟩)`, where `|10⟩` means orbital 0 occupied:

```text
even:  [[ μ, −Δ], [−Δ, −μ]]
odd:   [[ 0, −t], [−t,  0]]
```

Eigenvalues `±√(μ² + Δ²)` and `±t`. They match `spectrum` for `(t, Δ, μ) = (1, 0, 0)`, `(1, 1, 0)`, and `(1, 1, 1/2)`.

### 28. Kitaev Bogoliubov spectrum

The test builds this `2n × 2n` matrix itself. It does not call `spectrum` and rename the result.

```text
A_ii = −μ
A_{i,i+1} = A_{i+1,i} = −t
B_{i,i+1} = Δ
B_{i+1,i} = −Δ

[[ A,  B ],
 [ −B, −A ]]
```

Take the `n` algebraically largest eigenvalues `ε_k`. They are nonnegative. Every many-body energy is

```text
E(ν) = Σ_k (ν_k − 1/2) ε_k
```

For `n = 2, 3, 4` and `(t, Δ, μ)` in `{(1, 0, 0), (1, 1, 0), (1, 1, 0.5), (0.7, 1.3, 0.2)}`, the `2^n` values match `spectrum(kitaev_chain(...))` to `1e-8`. The `n = 2` case also matches the blocks from phase 27.

The `(n_i − 1/2)` convention already matches this occupation sum. Do not add or subtract `μ n / 2`.

### 29. Periodic tight binding

Add `tight_binding_ring(n, t=1.0, mapping="jordan_wigner")`. Same bonds as `tight_binding_chain`, plus the closing bond `−t (a†_{n−1} a_0 + h.c.)`.

Single-particle energies are `−2t cos(2π k / n)` for `k = 0, …, n−1`. Many-body energies are all fillings. Check `n = 3, 4, 5` at `t = 1` to `1e-8`.

`tight_binding_chain` stays open. Its existing tests stay green.

### 30. The closing bond is a Jordan–Wigner string

The periodic spin sum

```text
−(t/2) Σ_i (X_i X_{i+1} + Y_i Y_{i+1})
```

with `i + 1` taken mod `n`, is not the fermion ring. For `n = 4` and `t = 1` its spectrum differs from `tight_binding_ring`.

The fermion closing bond is the Pauli string

```text
−(t/2) ( X Z^{n−2} X + Y Z^{n−2} Y )
```

with the `Z`s on the middle qubits. For `n = 3, 4, 5` this term list equals `−t (a†_{n−1} a_0 + a†_0 a_{n−1})` after `simplify`, to `1e-12`. The spin ring that uses this closing bond, and the ordinary `XX + YY` bonds on the other links, matches `tight_binding_ring`.

### 31. The same ring in Bravyi–Kitaev

`tight_binding_ring(..., mapping="bravyi_kitaev")` matches the Jordan–Wigner spectrum to `1e-8` for `n = 3, 4, 5`. The default stays Jordan–Wigner. Existing tests are not rewritten around Bravyi–Kitaev.

### 32. One-particle RDM in either encoding

Give `one_particle_rdm` a `mapping` argument, default `"jordan_wigner"`. The Bravyi–Kitaev path uses `bravyi_kitaev_a` and `bravyi_kitaev_adag`.

On the vacuum, every natural occupation is 0. Apply `bravyi_kitaev_adag(0, 4)` to `basis(0, 4)`. Then `Tr γ = 1` and the largest occupation is 1.

Diagonalize `h2_sto3g_hamiltonian(1.401)` in each encoding, take each ground state, and compare natural occupations to `1e-8`. The two ground vectors live in different bases. Do not feed the Jordan–Wigner vector to the Bravyi–Kitaev RDM.

### 33. Sector tags in either encoding

Give `tag_sectors` a `mapping` argument, default `"jordan_wigner"`. The current behavior is the default.

Build the four-qubit state `a†_1 a†_0 |0⟩` with the Bravyi–Kitaev operators. The Bravyi–Kitaev tag is `(N, S_z) = (2, 0)`. The Jordan–Wigner tag of that same vector is not `(2, 0)`: its Jordan–Wigner particle number is 1. Reject a non-integer `⟨N⟩` as the existing function does.

## Time

### 34. Loschmidt echo

Add `loschmidt(H, state, time)`. Return `|⟨ψ(0)|ψ(t)⟩|²` from `evolve`.

For `H = (ω/2) Z`, `ω = 1`, and `|+⟩`, the value is `cos²(ω t / 2)`. Check `t = 0`, `t = π/ω`, and `t = 1`.

### 35. Spin echo

For `H = (ω/2) Z` and `|+⟩`, evolve for time `τ`, apply `X`, evolve for time `τ` again.

`X exp(−i H τ) X = exp(+i H τ)`, so the composed operator is `X`, and `X|+⟩ = |+⟩`. For `ω = 1.3` and `τ = 0.7`, the final state equals `|+⟩` up to a global phase, and `⟨X⟩ = 1`. Without the pulse, `⟨X⟩` at time `2τ` is `cos(ω · 2τ)`, which is not 1.

### 36. Ehrenfest

For `H = (ω/2) Z` and `|+⟩`, `⟨X(t)⟩ = cos(ω t)`, so its derivative is `−ω sin(ω t)`.

The same derivative is `i ⟨[H, X]⟩`. With `[Z, X] = 2i Y` this equals `−ω ⟨Y⟩`. A central difference of `expectation_label` along `evolve`, step `1e-6`, matches both expressions to `1e-8` at `t = 0`, `t = 0.3`, and `t = 1`, with `ω = 1.7`.

### 37. Two constant pieces

Add nothing new if `evolve` can be composed. The test is the check.

`ZI` and `IZ` commute, so `evolve(IZ, evolve(ZI, ψ, t), t)` matches `evolve(ZI + IZ, ψ, t)` for `t = 0.4` and `ψ = |+⟩⊗|+⟩`.

`ZI` and `XI` do not. The composed evolution differs from `evolve` of the sum at that same `t`. The state distance is above `1e-3`. Do not shrink `t` until the assertion passes.

### 38. Fourth-order product formula

Add `yoshida_unitary(H, dt)`. Let `S2` be `strang_unitary` and

```text
θ = 1 / (2 − 2^{1/3})
```

Then `S4(dt) = S2(θ dt) S2((1 − 2θ) dt) S2(θ dt)`. The middle step size is negative. `strang_unitary` must accept that.

Use the two-site transverse Ising chain, where `ZZ` does not commute with `X`. The same example as the first haul's Strang check. Measure the Frobenius distance of one step to the exact unitary. Halving `dt` cuts that distance by about thirty-two. Use several step sizes and a fixed tolerance around 5 on the log-log slope. Do not loosen the tolerance until the fit passes.

## Channels

### 39. Amplitude-damping Kraus map

Add `amplitude_damping(rho, p)` for one qubit, `0 ≤ p ≤ 1`.

```text
E0 = [[1, 0], [0, √(1−p)]]
E1 = [[0, √p], [0, 0]]
```

`E0† E0 + E1† E1 = I`. On `|1⟩⟨1|` the populations are `(p, 1−p)` for `(|0⟩, |1⟩)`. The purity of that state is `p² + (1−p)²`. `p = 0` leaves `|1⟩⟨1|`. `p = 1` sends it to `|0⟩⟨0|`.

### 40. Depolarizing channel

Add `depolarize(rho, p)` for one qubit,

```text
(1−p) ρ + (p/3) (XρX + YρY + ZρZ)
```

On `|0⟩⟨0|` the populations are `(1 − 2p/3, 2p/3)`. `p = 0` is the identity. `p = 3/4` is `I/2`. The Bloch vector is multiplied by `1 − 4p/3`.

### 41. Dephasing, closed form

Add `dephasing(rho, gamma, time)`. The equation is

```text
dρ/dt = γ (Z ρ Z − ρ)
```

so `ρ₀₁(t) = ρ₀₁(0) e^{−2γ t}`, and the diagonal is fixed. Check `|+⟩⟨+|` at `γ = 1`, `t = 0` and `t = ln 2`: the off-diagonal is `1/2` and then `1/4`. A central difference of the implemented map, step `1e-6`, matches the right-hand side to `1e-8`.

### 42. Amplitude damping, closed form

Add `amplitude_damping_evolve(rho, gamma, time)`. With `σ₋ = |0⟩⟨1|` and `σ₊ = |1⟩⟨0|`,

```text
dρ/dt = γ (σ₋ ρ σ₊ − ½ {σ₊ σ₋, ρ})
```

Then `ρ₁₁(t) = ρ₁₁(0) e^{−γ t}` and `ρ₀₁(t) = ρ₀₁(0) e^{−γ t / 2}`. On `|1⟩⟨1|` at `γ = 1`, `t = ln 2`, the excited population is `1/2`. A central difference matches the dissipator to `1e-8`, as in phase 41.

`p = 1 − e^{−γ t}` makes this map agree with phase 39 to `1e-12`.

### 43. Choi matrix

Add `choi(kraus_ops)` for a one-qubit channel, `Σ_{ij} |i⟩⟨j| ⊗ E(|i⟩⟨j|)`.

The identity channel has Choi eigenvalues `2, 0, 0, 0`. Amplitude damping at `p = 1/2` is positive semidefinite. The transpose map `E(ρ) = ρᵀ` is not completely positive: its Choi matrix is the swap, with eigenvalues `+1, +1, +1, −1`.

## Perturbation, geometry, and one derivative

### 44. Nondegenerate perturbation

`H = Z/2 + λ X`. The exact eigenvalues are `±√(1/4 + λ²)`. The ground state at `λ = 0` is `|1⟩`, with energy `−1/2`.

Second order in `λ` is `−1/2 − λ²`. At `λ = 10^{-3}` the exact ground energy and this polynomial differ by less than `10^{-11}`. The test writes the resolvent step: the only intermediate state is `|0⟩`, the gap is `−1`, and the numerator is `λ²`.

### 45. Degenerate perturbation

`H = XX + YY + ZZ + ε ZZ`, which is the XXZ dimer at `Δ = 1 + ε`.

Inside the triplet at energy `+1`, `ε ZZ` has expectations `+1` on `|00⟩`, `+1` on `|11⟩`, and `−1` on `(|01⟩ + |10⟩) / √2`. The first-order energies are `1+ε`, `1+ε`, and `1−ε`. They match `spectrum` of that XXZ dimer. The singlet is `−3 − ε` and is not part of the degenerate subspace.

### 46. Avoided crossing

Add nothing but the test, unless a two-by-two helper makes the assertion readable.

`H(z) = [[z, δ], [δ, −z]]` has eigenvalues `±√(z² + δ²)`. The gap is `2√(z² + δ²)`, with minimum `2|δ|` at `z = 0`. Check `δ = 0.1` at `z = 0` and at `z = 1`. At `δ = 0` the levels cross: eigenvalues `z` and `−z`.

### 47. Discrete Berry phase

`H(φ) = cos(φ) X + sin(φ) Y`. Its eigenvalues are `±1`.

Take the ground state on `φ_k = 2π k / N` for `N = 128`, `k = 0, …, N−1`, and the phase of `Π_k ⟨ψ_k | ψ_{k+1}⟩` with `ψ_N = ψ_0`. That phase is `π`, to `1e-2`.

A hand-written ground state is allowed in the test if its energy is `−1` at `φ = 0` and `φ = π/2` and it reproduces the same phase. Do not accept a fit.

### 48. Lanczos, exact when the Krylov space is full

Add `lanczos(H, start, steps)`. Reorthogonalize against every previous vector. If a `β` underflows, stop. The first diagonal entry equals `⟨start|H|start⟩`.

`|00⟩` is an eigenvector of `heisenberg_dimer` with energy `+1`. Lanczos from that ket stops at one step and returns `+1`.

From `(|000⟩ + |001⟩) / √2`, eight reorthogonalized steps on `transverse_ising(3, 1, 1)` return all eight eigenvalues, matching `spectrum` to `1e-8`. Do not assert a truncated ground-state approximation against a loose tolerance.

### 49. Hellmann–Feynman on the Kitaev chemical potential

`∂H/∂μ = −Σ_i (n_i − 1/2)`.

At `t = 0` and `Δ = 0` the vacuum has energy `μ n / 2` and derivative `n / 2`. Check `n = 2`, `μ = 0.3`.

A one-particle computational basis state has `⟨N⟩ = 1`, so the derivative is `n/2 − 1`. For `n = 2` that is 0, and the energy is 0 for every `μ`.

Do not difference the H₂ table.

## The scope line

### 50. What this haul added

When phases 1–49 are green, update `docs/scope.md` and the README so the new objects are described as present: Pauli permutation and Clifford conjugation, GHZ and W, negativity and entanglement of formation, Pauli measurement, Grover and the QFT, CHSH, parity and occupation blocks, the SSH chain, the Kitaev chain and its Bogoliubov spectrum, the periodic tight-binding ring and its Jordan–Wigner closing string, encoding-aware sector tags and one-particle RDMs, the spin echo, the fourth-order product formula, the qubit channels, and the Kitaev chemical-potential derivative.

Leave these named as not done:

- A derivative of the H₂ curve.
- Any molecule beyond H₂ and HeH⁺.
- Exact diagonalization above eight qubits.

The nine-qubit Pauli expectation from phase 2 is not that third item.

## Done

After phase 50 the lab is still a state, a Hamiltonian, and a spectrum, now with the questions those three can answer exactly. No new dependency. No third molecule. The eight-qubit diagonalization limit is where it was.
