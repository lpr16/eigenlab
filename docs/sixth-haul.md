# Sixth haul

The first five hauls are in the tree. The lab has three fermion encodings, tapers a qubit when a single `Z` is conserved, and checks stabilizer codes with matrices. This haul does that algebra over GF(2): general symmetry tapering, stabilizer tableaux, the Clifford group and its exact averages, and magic. Tableaux and symmetric-subspace blocks scale past eight qubits without diagonalizing anything. The haul then adds collective spins and squeezing, Bell and contextuality inequalities by enumeration, a Floquet spectrum, the Heisenberg limit of the Hubbard chain, and a second basis for hydrogen. Thirty-seven phases, in order. Each one is a single change set with a closed-form or cited check.

Do not start phase k+1 until phase k is green under `pytest`.

This file is the spec for an agent, including Google Antigravity. A workflow file there is capped at 12,000 characters, so the spec stays here. Point `/goal` at one phase. If you are asked to do the whole haul, finish phase k, leave `pytest` green, then start phase k+1. Do not implement two phases behind one test run. Do not bump the package version. Do not commit or push unless the person running the agent says to. Completion of a phase is the new assertions passing, not a summary of intent.

## Contract

Stay inside this stack: Python 3.11, NumPy, pytest, and the standard library. Do not add SciPy, Qiskit, Stim, PennyLane, OpenFermion, PySCF, QuTiP, galois, or any other quantum or finite-field package. GF(2) elimination is written by hand on `uint8` arrays. Never send a GF(2) matrix through `numpy.linalg`.

`pauli("XZ")` means qubit 0 is the leftmost character and the most significant bit of `basis(index, qubits)`. Orbital 0 is that same qubit. `spatial_to_spin_orbital` interleaves spins as `(↑, ↓, ↑, ↓, …)`. Jordan–Wigner stays the default. A qubit with a level splitting is `H = −(ω/2) Z`. Do not flip any of these.

A Pauli string as bits is `(x | z)`: `I = (0, 0)`, `X = (1, 0)`, `Z = (0, 1)`, `Y = (1, 1)`, one pair per qubit in label order.

Size limits:

- A many-body matrix has dimension at most 256: eight qubits, or an oscillator times a qubit with `2d ≤ 256`.
- A Liouvillian acts on at most four qubits.
- A stabilizer tableau is a bit matrix, not a many-body matrix, and may go to `n = 128` qubits.
- A collective-spin matrix on the symmetric subspace has dimension `N + 1` and may go to `N = 64`.
- A `2n × 2n` Bogoliubov matrix may go to `n = 64`.
- An exact Clifford average enumerates at most `24³` sequences.
- Do not diagonalize anything above eight qubits, and do not add a matrix-free solver to get around that.

A test number is one of three things: a few lines of algebra written next to the assertion, a value this repo already cites, or a value from a citation named in the test module with its table or page. Do not invent a decimal to make a test pass. Do not loosen a tolerance until a fit succeeds. Where a phase calls an inequality a physics claim, it is not a target: if it fails, stop and report. Do not move a parameter until it passes.

Nothing in this haul draws a random number. Where a measurement outcome is random, the caller names the outcome and gets its probability back. Averages over the Clifford group are exact sums, not samples.

Out of this haul: a fourth molecule, p or higher basis functions, 6-31G for any atom but hydrogen, open-shell Hartree–Fock, variational algorithms, shot noise and sampling, hardware, non-Clifford gates inside a tableau, tapering by a symmetry that is not a product of `Z`s, and exact diagonalization above eight qubits.

## Symmetries over GF(2)

Put the bit machinery in a new module, `src/eigenlab/stabilizer.py`. Tapering goes in `hamiltonian.py`, next to `taper`.

### 1. Paulis as bit vectors

Add `pauli_to_bits(label)`, which returns `(x, z)` as `uint8` arrays, and `bits_to_pauli(x, z)`, which inverts it.

Two labels commute exactly when `x₁·z₂ + z₁·x₂ = 0 (mod 2)`. That matches `pauli_comm` for every pair of length-2 labels (256 pairs) and every pair of length-3 labels (4096 pairs). Round-trips are exact.

Multiplying two Paulis adds their bit vectors mod 2. The phase is not carried by the bits; `pauli_mult` still owns it. For every pair of length-2 labels, `bits_to_pauli` of the sum is the label `pauli_mult` returns.

### 2. Linear algebra over GF(2)

Add `gf2_rank(M)` and `gf2_nullspace(M)`. The nullspace is returned as rows `v` with `M v = 0 (mod 2)`. Implement them with Gaussian elimination.

- `gf2_rank(I_n) = n`.
- The all-ones `3 × 3` matrix has rank 1.
- The Hamming `[7, 4]` parity-check matrix, with rows `0001111`, `0110011`, `1010101`, has rank 3. Its nullspace has 4 rows, each satisfying `M v = 0`, and together they have rank 4.
- Over the reals that same matrix also has rank 3, so the test adds a case where the two answers differ. The rows `110`, `011`, `101` have GF(2) rank 2, since they sum to zero mod 2, but real rank 3.

### 3. The Z₂ symmetries of a Hamiltonian

Add `z2_symmetries(H)`. It returns independent labels made only of `I` and `Z` that commute with every term of `H`, with the identity excluded.

A `Z`-type string with bit vector `a` commutes with a term whose x-part is `x_t` exactly when `a · x_t = 0 (mod 2)`. So the symmetries are the GF(2) nullspace of the matrix whose rows are the terms' x-parts. Write that derivation in the test.

- Every returned label commutes with every term (`pauli_comm`) and with `H.to_matrix()` to `1e-12`.
- Jordan–Wigner H₂ at `R = 1.401` has exactly 3 generators. Bravyi, Gambetta, Mezzacapo and Temme, arXiv:1701.08213, taper H₂ to one qubit.
- The group they generate (`stabilizer_group`) contains `ZIZI`, `IZIZ` and `ZZII`. These are the parities of `N_↑ = n₀ + n₂`, of `N_↓ = n₁ + n₃`, and of the σ_g occupation `n₀ + n₁`, which the double excitation `σ_g² ↔ σ_u²` changes by two. Write that in the test.

### 4. One Clifford per symmetry

Add `z2_taper(H, signs, generators=None)`. Bring the generators to reduced row echelon form over GF(2). Then generator `i` has a pivot qubit `q_i` where it alone has a `Z`, so `X_{q_i}` anticommutes with generator `i` and commutes with every other generator.

Let `U_i = (X_{q_i} + τ_i)/√2`. That is Hermitian and unitary because the two terms anticommute. `U_i τ_i U_i = X_{q_i}`. Conjugate `H` by every `U_i`. Then each pivot qubit carries only `I` or `X`. Replace `X_{q_i}` by its sign and drop the qubit.

The test builds `U = Π U_i` as a matrix for the H₂ case and checks:

- `U` is unitary.
- `U τ_i U† = X_{q_i}`.
- `U H U†` has only `I` and `X` on every pivot qubit.
- The term-list result of `z2_taper` matches the matrix result for every sign choice.

### 5. H₂ on one qubit

For Jordan–Wigner H₂ at `R = 1.0, 1.401, 2.0, 3.0`:

- The eight one-qubit tapered spectra, over all sign choices, together equal the sixteen-level four-qubit spectrum to `1e-10`, multiplicities included.
- The lowest tapered level is the four-qubit ground energy to `1e-10`. At `R = 1.401` it is the cited `−1.13727` to `1e-5`, and it equals the ground energy of the fifth haul's two-qubit parity taper to `1e-10`.

### 6. H₄, tapered

Take Jordan–Wigner `h4_sto3g_hamiltonian(1.401, 2.0)` in Löwdin orbitals.

`z2_symmetries` finds at least 2 generators. The parities of `N_↑` (Z on the even qubits) and of `N_↓` (Z on the odd qubits) are always there.

Taper all of them. The union of the tapered spectra equals the 256-level spectrum to `1e-8`, multiplicities included.

## Stabilizer tableaux

### 7. The tableau

Add a `Tableau(n)` class following Aaronson and Gottesman, Phys. Rev. A 70, 052328 (2004).

- Rows `0 … n−1` are destabilizers and rows `n … 2n−1` are stabilizers. Each row has `x` and `z` bits plus a sign bit `r`.
- `Tableau(n)` starts in `|0…0⟩`.
- The gates are `h(q)`, `s(q)` and `cnot(c, t)`, with the update rules of that paper.
- `stabilizers()` returns signed labels.

The test lists six fixed circuits on `n = 2 … 6`. For each one, it builds the same circuit with matrices on a state vector and checks `P|ψ⟩ = sign · |ψ⟩` for every returned stabilizer `P`. Destabilizer `i` anticommutes with stabilizer `i` and commutes with every other stabilizer.

### 8. Measurement without sampling

Add `measure(q, outcome=None)` for a `Z` measurement. It returns `(outcome, probability)`.

- If the outcome is deterministic, the probability is 1. Passing the opposite outcome raises.
- If it is random, the probability is `1/2`, the caller must pass the outcome, and leaving it as `None` raises.

The tableau then holds the post-measurement state. For the circuits of phase 7 and every qubit, the probability and the post-measurement stabilizers match `pauli_probabilities` and `post_measure` on the state vector.

### 9. Expectations from the group

Add `Tableau.expectation(label)`, which returns `−1`, `0` or `+1`.

- It is `0` if the label anticommutes with some stabilizer.
- Otherwise the label is `±` a product of stabilizers, and the destabilizers say which product. Read the sign off.

For the circuits of phase 7 with `n ≤ 4`, every label of length `n` (`4ⁿ` of them) matches `expectation_label` on the state vector exactly.

### 10. Sixty-four qubits

Apply `h(0)`, then `cnot(0, k)` for `k = 1 … 63`.

- `⟨Z₀ Z_k⟩ = 1` for every `k`.
- `⟨X^{⊗64}⟩ = 1`.
- `⟨Y Y X^{⊗62}⟩ = −1`. A product of `X`s and `Y`s on GHZ has expectation `(−1)^{#Y/2}` when `#Y` is even. Write that in the test.
- `⟨Z₀⟩ = 0`.

Measure qubit 0 with outcome 1. Its probability is `1/2`, and every other qubit's `Z` measurement is then deterministic with outcome 1. No 64-qubit matrix is built.

### 11. The toric code on larger tori

Generalize the fifth haul's edges to an `L × L` torus, with coordinates mod `L`. Horizontal edge `h(x, y) = 2(Ly + x)` and vertical edge `v(x, y) = 2(Ly + x) + 1`. Stars and plaquettes follow the same rules. At `L = 2` the labels equal `toric_code_stabilizers()`.

For `L = 2, 3, 4, 5`, with up to 50 qubits:

- The `2L²` generators have GF(2) rank `2L² − 2`, which leaves 2 logical qubits.
- The straight loop of `Z` on `h(x, 0)` for every `x` commutes with every generator. Appending it raises the rank, so it is not in the group. Its weight is `L`.
- At `L = 3`, no label of weight 1 or 2 commutes with every generator without being in the group. Check this in bits, not matrices: there are `54 + 1377` labels. So the distance is 3.

### 12. Teleportation by tableau

Run the fifth haul's teleportation circuit on a tableau, for the six stabilizer inputs `|0⟩, |1⟩, |+⟩, |−⟩, |+i⟩, |−i⟩`. Prepare them with `h` and `s`; `X = H S S H`.

For each of the four forced outcomes `(m₀, m₁)`:

- Each outcome has probability `1/4`.
- After the corrections `Z^{m₀} X^{m₁}`, with `Z = S S`, qubit 2's stabilizer is the input's stabilizer with its sign.

## The Clifford group

Put this in a new module, `src/eigenlab/clifford.py`.

### 13. Twenty-four elements

Add `one_qubit_cliffords()`: the closure of `{H, S}` under multiplication, modulo global phase. Fix the phase so the first nonzero entry is real and positive.

- There are 24 elements.
- Each conjugates `X`, `Y` and `Z` to signed Paulis.
- The map `C ↦ (C X C†, C Z C†)` is injective, and `24 = 6 × 4`: `X` can go to any of 6 signed Paulis, and `Z` then to any of the 4 that anticommute with it. Write that in the test.

### 14. A unitary 2-design

The second frame potential of a group `G` is `(1/|G|) ∑_{C ∈ G} |Tr C|⁴`. Haar measure gives 2 for `d ≥ 2` (Gross, Audenaert and Eisert, J. Math. Phys. 48, 052104 (2007)).

Count by class:

- the identity gives `16`
- the 3 Paulis are traceless
- the 8 order-3 rotations have `|Tr| = 1`, giving `8`
- the 6 quarter turns have `|Tr| = √2`, giving `24`
- the 6 edge half-turns are traceless

That is `48/24 = 2`. Write the count in the test. The one-qubit Pauli group, modulo phase, gives 4 and is not a 2-design.

The six stabilizer states are a state 2-design: `(1/36) ∑_{ij} |⟨ψ_i|ψ_j⟩|⁴ = 1/3 = 2/(d(d+1))`. The pairs contribute 6 identical pairs at 1, 6 antipodal pairs at 0, and 24 pairs at `1/4`.

### 15. Twirling

Add `twirl(channel)`, which averages `C† channel(C ρ C†) C` over the 24 elements.

For amplitude damping, the entanglement fidelity is `F_e = ∑_k |Tr E_k|²/d² = (1 + √(1−p))²/4`. The twirl is depolarizing, `ρ ↦ λρ + (1−λ) I/2` with `λ = (4F_e − 1)/3`, and that equals `depolarize(ρ, 3(1−λ)/4)`.

The average fidelity over the six stabilizer states is `(2F_e + 1)/3 = (1 + λ)/2`. A 2-design average is the Haar average.

Check `p = 0, 0.3, 1` to `1e-12`. At `p = 1`, every state goes to `|0⟩`, and the average of `|⟨ψ|0⟩|²` over the six states is `1/2`.

### 16. Benchmarking, averaged exactly

Apply `m` uniformly chosen Cliffords, each followed by amplitude damping `Λ`, then the noiseless inverse of their product, and measure `|0⟩`.

Write `D_i = C_i ⋯ C_1`. The sequence is `∏ D_i† Λ D_i`, and the `D_i` are independent and uniform, so every factor is the twirl of phase 15. Hence the survival probability is

```text
F(m) = (1 + λᵐ)/2
```

Write that derivation in the test. Enumerate all `24ᵐ` sequences for `m = 1, 2, 3`, and match it to `1e-12` at `p = 0.1` and `p = 0.4`. Nothing is sampled.

## Magic

### 17. T is not Clifford

`T = diag(1, e^{iπ/4})`. Then `T X T† = (X + Y)/√2`, which is not a signed Pauli. Write the algebra in the test, using `|0⟩⟨1| = (X + iY)/2`.

`(X + Y)/√2` is itself a Clifford: it sends `X → Y`, `Y → X`, `Z → −Z`. So `T` conjugates every Pauli into the Clifford group, which places it in the third level of the Clifford hierarchy.

### 18. Stabilizer Rényi entropy

Add `stabilizer_renyi_entropy(state)`, defined as

```text
M₂ = −log₂( (1/d) ∑_P ⟨P⟩⁴ )
```

with the sum over the `4ⁿ` unsigned Pauli strings (Leone, Oliviero and Hamma, Phys. Rev. Lett. 128, 050402 (2022)).

- It is 0 on the six one-qubit stabilizer states, and on the state vectors of the phase-7 circuits with `n ≤ 4`.
- `|T⟩ = T|+⟩` has `⟨X⟩ = ⟨Y⟩ = 1/√2` and `⟨Z⟩ = 0`, so the sum is `1 + 1/4 + 1/4 = 3/2` and `M₂ = log₂(4/3)`.
- `|T⟩^{⊗k}` has `k log₂(4/3)` for `k = 1 … 4`.
- Each of the 24 one-qubit Cliffords leaves `M₂` unchanged on a fixed non-stabilizer state.

### 19. The most magic qubit

On one qubit, `∑_P ⟨P⟩⁴ = 1 + r_x⁴ + r_y⁴ + r_z⁴`. Cauchy–Schwarz gives `∑ r_i⁴ ≥ (∑ r_i²)²/3 = 1/3` on pure states, with equality at `|r_x| = |r_y| = |r_z|`. So `M₂ ≤ log₂(3/2)`. Write the algebra in the test.

The state with Bloch vector `(1, 1, 1)/√3` attains it to `1e-12`. On a `181 × 361` grid in `(θ, φ)`, no pure state exceeds `log₂(3/2) + 1e-12`.

## Collective spin

Put this in a new module, `src/eigenlab/collective.py`.

### 20. The symmetric subspace

Add these:

- `collective_spin(n)`, which returns `J_x, J_y, J_z = ∑ σ/2` as term lists.
- `spin_matrices(j)`, the `(2j+1)`-dimensional matrices with `J_z = diag(j, …, −j)` and `J_+` elements `√(j(j+1) − m(m+1))`.
- `dicke(n, k)`, the normalized sum of the weight-`k` basis states.

Order the Dicke states as `m = n/2 − k`. Let `V` have columns `dicke(n, k)` for `k = 0 … n`. Then `V† J_a V = spin_matrices(n/2)` for every `a` and every `n ≤ 8`, to `1e-12`.

### 21. Dicke states

`dicke(3, 1)` is `w_state(3)`.

Cutting after `ℓ` qubits, the Schmidt probabilities are hypergeometric:

```text
λ_j = C(ℓ, j) C(n−ℓ, k−j) / C(n, k)
```

The entropy built from them equals `bipartite_entropy` to `1e-10` for every `n ≤ 8`, every `k` and every cut.

### 22. Fisher information of Dicke states

In `|j, m⟩`, `Var(J_x) = (j(j+1) − m²)/2`, so `F = 2(j(j+1) − m²)`. Write that in the test.

- `qfi(dicke(n, k), J_x)` matches for `n ≤ 8` and every `k`.
- `spin_matrices` reproduces it for `n = 16, 32, 64`.
- At `k = n/2`, `F = n(n+2)/2`. That beats the product state's `n` for every `n ≥ 1` and falls short of GHZ's `n²` for every `n ≥ 3`, since `n + 2 < 2n` there.

### 23. One-axis twisting

Let `H = χ J_z²`, and start from `|+⟩^{⊗n}`, the coherent spin state along `x`.

Only the terms of `H` containing spin `k` touch it, and they are `(χ/2) σ_z^k ∑_{j≠k} σ_z^j`. Raising spin `k` therefore picks up a phase from each other spin, and in `|+⟩` each one averages to `cos(χt)`:

```text
⟨J_x(t)⟩ = (n/2) cos^{n−1}(χt),   ⟨J_y⟩ = ⟨J_z⟩ = 0
```

This is Kitagawa and Ueda, Phys. Rev. A 47, 5138 (1993). Write the derivation in the test.

- `evolve` on qubits matches for `n ≤ 8`, with `χ = 0.1` and `t` in `{0.5, 2, 7}`, to `1e-10`.
- The symmetric subspace matches for `n = 16, 32, 64`.

### 24. Squeezing bounds the Fisher information

Let `V_min` be the smaller eigenvalue of the symmetrized `y–z` covariance matrix, and `s` its direction. The squeezing parameter is

```text
ξ² = n V_min / ⟨J_x⟩²
```

- At `t = 0`, `ξ² = 1`: `V_min = n/4` and `⟨J_x⟩ = n/2`.
- For the generator `J_⊥` along `x̂ × s`, Pezzé and Smerzi, Phys. Rev. Lett. 102, 100401 (2009), give `F ≥ ⟨J_x⟩²/V_min = n/ξ²`. On a pure state `F = 4 Var(J_⊥)`. Check the bound for `n = 8` and `n = 64` at five times.
- Physics claim: at `n = 64` and `χt = 0.05`, `ξ² < 1`.

## Nonlocality and contextuality

Put this in a new module, `src/eigenlab/nonlocality.py`.

### 25. Mermin, three qubits

`M = XXX − XYY − YXY − YYX`.

On `ghz(3)`, `XXX = +1`, and each string with two `Y`s gives `−1`: `XYY|000⟩ = |1⟩(i|1⟩)(i|1⟩) = −|111⟩`. So `⟨M⟩ = 4`, which is also the largest eigenvalue of `M`.

Assigning `x_k, y_k ∈ {±1}` to every local setting, the largest value of `x₁x₂x₃ − x₁y₂y₃ − y₁x₂y₃ − y₁y₂x₃` is 2. Find it by enumerating all 64 assignments.

### 26. Mermin, by enumeration

Add `mermin_operator(n) = Re ∏_k (X_k + iY_k)`. As a term list, it is every string of `X`s and `Y`s with an even number of `Y`s, signed `(−1)^{#Y/2}`.

`∏ (X_k + iY_k) = 2ⁿ |0…0⟩⟨1…1|`, so `M_n` has eigenvalues `±2^{n−1}` and 0, and `⟨ghz(n)|M_n|ghz(n)⟩ = 2^{n−1}`. Write that in the test.

The local deterministic maximum, found by enumerating all `4ⁿ` assignments for `n = 3 … 8`, equals `2^{n/2}` for even `n` and `2^{(n−1)/2}` for odd `n`. Each factor `x + iy` has modulus `√2` and phase an odd multiple of `π/4`. Write that in the test. The ratio of the quantum value to that maximum grows with `n`.

### 27. The Peres–Mermin square

```text
XI  IX  XX
IZ  ZI  ZZ
XZ  ZX  YY
```

- Every row and every column is three mutually commuting labels.
- Each row multiplies to `+I`.
- The columns multiply to `+I`, `+I` and `−I`. The last column is `XZY ⊗ XZY = (−i)² I`.
- None of the `2⁹` assignments of `±1` to the nine cells reproduces all six products. This holds for every state.

### 28. CHSH with noise

For two qubits, let `T_ij = Tr(ρ σ_i ⊗ σ_j)`, and let `M(ρ)` be the sum of the two largest eigenvalues of `TᵀT`. The largest CHSH value is `2√M(ρ)` (Horodecki, Horodecki and Horodecki, Phys. Lett. A 200, 340 (1995)).

For `werner_state(p)`, `T = p diag(1, −1, 1)`, so the largest value is `2√2 p`.

- `chsh_operator` attains it on the Werner state.
- No setting on a `37 × 73` grid of measurement directions exceeds it by more than `1e-12`.
- For `p` in `(1/3, 1/√2]`, the state is entangled (`concurrence > 0`) and violates nothing. Assert both at `p = 0.5`.

## Time

### 29. Floquet quasienergies

Take the fifth haul's rotating drive, `H(t) = −(ω₀/2) Z + (Ω/2)(cos ωt X − sin ωt Y)`, with period `T = 2π/ω`.

In the frame `R(t) = e^{iωtZ/2}`, `U(t) = R(t) e^{−iH_eff t}`, and `R(T) = e^{iπZ} = −I`. Hence

```text
U(T) = −e^{−i H_eff T},   quasienergies ε = ω/2 ± √(Ω² + δ²)/2   (mod ω)
```

Write that in the test.

At `ω₀ = 1`, `Ω = 0.2`, `ω = 1.1`, build `U(T)` by applying `evolve_driven` to both basis states. Double the number of steps until the matrix changes by less than `1e-8`. `U(T)` then matches `−e^{−iH_eff T}` to `1e-6`, and its eigenphases match the quasienergies to `1e-6`.

## Hubbard to Heisenberg

### 30. Hubbard chain

Add `hubbard_chain(n_sites, t=1.0, U=0.0, mapping="jordan_wigner")`. It is an open chain with orbitals `2i` (↑) and `2i+1` (↓), and `n_sites ≤ 4`.

- At `n_sites = 2` it equals `hubbard_dimer` after `simplify`, to `1e-12`.
- At `U = 0` its spectrum is every filling of two copies of the open tight-binding single-particle energies, to `1e-10`.

### 31. The Heisenberg limit

At half filling and large `U`, second-order perturbation theory gives

```text
H_eff = J ∑_bonds (S_i · S_j − 1/4),   J = 4t²/U
```

With `S · S = σ · σ / 4`, this is `(J/4)(H_σ − (n−1))`, where `H_σ = ∑_bonds (XX + YY + ZZ)`.

At `n = 2` this gives `−J = −4t²/U`. Expanding `(U − √(U² + 16t²))/2 = −4t²/U + 16t⁴/U³ + …` confirms it. Write that in the test.

For `n_sites = 4` in the `N = 4`, `S_z = 0` block, let `E₀(U)` be the Hubbard ground energy. Its difference from `(J/4)(min spec H_σ − 3)` falls as `U⁻³`. Fit the log-log slope over `U = 20, 40, 80, 160` with `t = 1`: it is `−3` within a fixed tolerance of `0.05`.

### 32. The whole low band

In the same block, the six lowest Hubbard levels map onto the six levels of `(J/4)(H_σ − 3)` restricted to `S_z = 0` (dimension `C(4, 2) = 6`), in order. The largest of the six differences also falls as `U⁻³`, with the same tolerance.

At `U = 20`, the seventh level is separated from the sixth by more than `U/2`.

## A second basis for hydrogen

### 33. 6-31G

Generalize `ao_integrals` to take a list of contracted s functions. The STO-3G entry point stays, and returns the same arrays to `1e-14`.

Add 6-31G for hydrogen, from Ditchfield, Hehre and Pople, J. Chem. Phys. 54, 724 (1971):

- inner function: exponents `18.7311370, 2.8253937, 0.6401217` with coefficients `0.03349460, 0.23472695, 0.81375733`
- outer function: exponent `0.1612778` with coefficient `1.0`

Each contracted function is normalized to `1e-6`.

### 34. The hydrogen atom in 6-31G

Solve the `2 × 2` problem `h c = E S c` through `lowdin`.

`E > −1/2` strictly, by the variational principle against the exact `−1/2`. Write that in the test.

Physics claim: `E < H_ATOM_STO3G_ENERGY`. The two spans are not nested, so this is not a theorem. If a cited 6-31G hydrogen energy is at hand, compare it, and name the source in the test.

### 35. H₂ in 6-31G

Add `h2_631g_integrals(R)` and `h2_631g_hamiltonian(R, mapping="jordan_wigner")`. They give four spatial orbitals and eight qubits. Work in the `N = 2` block, which has dimension `C(8, 2) = 28`.

- The Hamiltonian commutes with `total_number_operator(8)` and `total_spin_z(4)`.
- The full-CI energy is the same in Löwdin orbitals and RHF orbitals, to `1e-8`.
- The RHF energy equals `⟨Φ|H|Φ⟩` on `basis(0b11000000, 8)`, to `1e-10`.
- At `R = 20`, full CI is `2E` with `E` from phase 34, to `1e-6`.
- The full-CI gradient changes sign on `[1.0, 2.0]`. Bisect it as in the integral haul.
- Physics claim: 6-31G full CI at `R = 1.401` lies below the cited STO-3G `−1.13727`.

### 36. 6-31G H₂, tapered

In RHF orbitals, every canonical orbital of H₂ is gerade or ungerade. So `z2_symmetries` finds at least 3 generators: the parities of `N_↑`, of `N_↓`, and of the ungerade occupation. Write that in the test.

Taper them. Each sector acts on at most 5 qubits. The union of the tapered spectra equals the 256-level spectrum to `1e-8`, and the `N = 2` ground energy appears in exactly one sector.

## The scope line

### 37. What this haul added

When phases 1–36 are green, update `docs/scope.md` and the README so these are described as present:

- Pauli strings as bit vectors, GF(2) rank and nullspace, and Z₂ symmetry tapering, with H₂ on one qubit
- stabilizer tableaux with exact measurement and expectations, a 64-qubit GHZ state, and the toric code on tori up to `5 × 5`
- the one-qubit Clifford group, its 2-design checks, twirling, and exactly averaged randomized benchmarking
- the T gate's place in the Clifford hierarchy and the stabilizer Rényi entropy
- collective spin, Dicke states, one-axis twisting, and the squeezing bound on the Fisher information
- the Mermin and Peres–Mermin inequalities by enumeration, and the Horodecki CHSH criterion
- Floquet quasienergies of the rotating drive
- the Heisenberg limit of the Hubbard chain
- 6-31G for hydrogen, and H₂ in eight qubits in that basis

Change the not-done list to:

- Any molecule beyond H₂, HeH⁺, and H₄.
- Any basis with p or higher functions, and 6-31G for any atom but hydrogen.
- Exact diagonalization above eight qubits.

A tableau on 64 qubits and a symmetric-subspace block at `N = 64` are not exact diagonalization above eight qubits, and they do not close the third item.

## Done

After phase 37 the lab is still a state, a Hamiltonian, and a spectrum. It now also does algebra over GF(2), reaches past eight qubits without diagonalizing anything there, and averages over the Clifford group exactly. No new dependency. The eight-qubit diagonalization limit is where it was.
