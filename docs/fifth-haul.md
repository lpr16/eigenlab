# Fifth haul

The first four hauls are in the tree. The molecules are functions, the codes are checked, and the free fermions have a partition function and an entropy. This haul adds what still fits inside eight qubits with an exact answer: a third fermion encoding and the qubits it lets you drop, translation symmetry and the magnons it labels, the toric code on the smallest torus, teleportation and phase estimation, quantum Fisher information, a time-dependent drive, the Lindblad equation, and a qubit coupled to an oscillator. Thirty-nine phases, in order. Each one is a single change set with a closed-form or cited check.

Do not start phase k+1 until phase k is green under `pytest`.

This file is the spec for an agent, including Google Antigravity. Point `/goal` at one phase. If you are asked to do the whole haul, finish phase k, leave `pytest` green, then start phase k+1. Do not implement two phases behind one test run. Do not bump the package version. Do not commit or push unless the person running the agent says to. Completion of a phase is the new assertions passing, not a summary of intent.

## Contract

Stay inside this stack: Python 3.11, NumPy, pytest, and the standard library. Do not add SciPy, Qiskit, PennyLane, OpenFermion, PySCF, QuTiP, or any other quantum package. The matrix exponential in phase 29 is written by hand.

`pauli("XZ")` means qubit 0 is the leftmost character and the most significant bit of `basis(index, qubits)`. Orbital 0 is that same qubit. `spatial_to_spin_orbital` interleaves spins as `(↑, ↓, ↑, ↓, …)`. Jordan–Wigner stays the default. Do not flip any of these.

A qubit with a level splitting is written `H = −(ω/2) Z`. Then `|0⟩` is the ground state and `|1⟩` the excited state, which matches `amplitude_damping`, where `|1⟩` decays to `|0⟩`, and `σ₋ = |0⟩⟨1|` lowers the energy.

Size limits:

- A many-body matrix has dimension at most 256: eight qubits, or an oscillator times a qubit with `2d ≤ 256`.
- A Liouvillian acts on at most four qubits, so it is at most `256 × 256`.
- A `2n × 2n` single-particle Bogoliubov matrix is not a many-body matrix and may go to `n = 64`.
- Do not diagonalize anything above eight qubits, and do not add a matrix-free solver to get around that.

A test number is one of three things: a few lines of algebra written next to the assertion, a value this repo already cites, or a value from a citation named in the test module with its table or page. Do not invent a decimal to make a test pass. Do not loosen a tolerance until a fit succeeds.

Out of this haul: a fourth molecule, any basis beyond STO-3G s functions, open-shell Hartree–Fock, variational algorithms, shot noise and sampling, hardware, stabilizer tableaux, exact diagonalization above eight qubits, and tapering by a symmetry that is not a single-qubit `Z`.

## A third encoding

### 1. Parity operators

Add `parity_a` and `parity_adag`. They return a `Hamiltonian` term list.

Qubit `j` of the parity encoding stores `p_j = x_0 + x_1 + … + x_j` over GF(2). The map from `x` to `p` is the CNOT circuit `CNOT(0→1), CNOT(1→2), …, CNOT(n−2→n−1)`, in that order. Conjugate the Jordan–Wigner operators through it with the same `_conjugate_by_cnots` that builds Bravyi–Kitaev. The order matters: the reversed sequence gives the right answer for `n ≤ 2` and the wrong one from `n = 3` on.

Test oracle, for `n ≤ 4`: let `Π` be the permutation `|x⟩ → |p⟩`. Then `parity_a(p, n).to_matrix()` equals `Π @ jordan_wigner_a(p, n).to_matrix() @ Π.T` for every orbital.

For every `n ≤ 5`: `{a_i, a†_j} = δ_ij I`, `{a_i, a_j} = 0`, `a_i² = 0`. The vacuum is still `basis(0, n)`.

### 2. Parity wherever a mapping is accepted

Add `"parity"` (alias `"p"`) to `_ladder_operators`, so every builder that takes `mapping=` accepts it. `total_spin_squared` does not take `mapping=` yet; give it one. An unknown mapping still raises.

With the parity mapping:

- `tight_binding_chain` at `n = 3, 4, 5` has the Jordan–Wigner eigenvalues to `1e-8`.
- `hubbard_dimer` at the `(t, U)` pairs in `test_hubbard_dimer` has the full Jordan–Wigner spectra, degeneracies included.
- `h2_sto3g_hamiltonian(1.401, mapping="parity")` and `heh_sto3g_hamiltonian(mapping="parity")` have the Jordan–Wigner ground energies to `1e-8`.
- `a†_p` applied to the vacuum has `⟨N⟩ = 1` under `total_number_operator(n, mapping="parity")`.

### 3. Qubits that only see Z

Add `z_symmetry_qubits(H)`: the sorted qubits on which every term of `H` acts as `I` or `Z`. On such a qubit `Z_q` commutes with every term, hence with `H`.

- Parity-mapped H₂ at `R = 1.401`: `[1, 3]`. Qubit 3 stores `N mod 2`. Qubit 1 stores the occupation parity of spatial orbital 0, the σ_g orbital; inversion symmetry conserves it.
- Jordan–Wigner H₂: `[]`. Its double-excitation terms act on all four qubits with `X` and `Y`.
- Parity-mapped HeH⁺: `[3]`. HeH⁺ has no inversion centre, so only `N mod 2` survives.

### 4. H₂ on two qubits

Add `taper(H, qubits, signs)`. On each listed qubit, replace `Z` by its sign (`+1` or `−1`) and `I` by 1, then delete the qubit. Combine equal labels. The result acts on `n − len(qubits)` qubits.

`Z = +1` is parity bit 0, an even count.

For H₂ at `R = 1.0, 1.401, 2.0, 3.0`:

- The four tapered two-qubit spectra, for signs in `{±1}²`, together equal the sixteen-level four-qubit spectrum to `1e-10`, multiplicities included.
- The ground state sits in sector `(+1, +1)`: an even electron count and an even σ_g occupation. Its energy is the four-qubit ground energy to `1e-10`, and at `R = 1.401` the cited `−1.13727` to `1e-5`.

### 5. HeH⁺ on three qubits

Taper parity-mapped HeH⁺ on qubit 3.

- Sign `+1` (even `N`) has lowest energy `−2.880712` to `1e-5`, the cation value already cited in `test_heh_sto3g`. The even sector also holds `N = 0` and `N = 4`. Both lie higher; tag them with `tag_sectors` on the four-qubit eigenvectors rather than assuming.
- Sign `−1` (odd `N`) has lowest energy `−2.922682` to `1e-5`, the `N = 3` value already named in that test's docstring.

## The Ising chain, by hand

### 6. Two-site modes

For `n = 2`, `transverse_ising` has the energies `±√(J² + 4h²)` and `±J`. The ground state is `E(0, 0) = −(ε₁ + ε₂)/2`, and the next state up is `E(1, 0) = (ε₁ − ε₂)/2 = −J`. Hence

```text
ε_{1,2} = √(J² + 4h²) ∓ J
```

Write that algebra in the test. The `n` nonnegative eigenvalues of `transverse_ising_bdg_matrix(2, J, h)` equal these for `(J, h)` in `{(1, 0.5), (0.7, 1.3), (1, 1)}` to `1e-12`. At `h = 0`, `ε₁ = 0`: a zero mode, and a doubly degenerate ground state.

### 7. The critical chain

At `J = h = 1` the open chain's modes are exact:

```text
ε_m = 4 sin((2m − 1) π / (2(2n + 1))),   m = 1, …, n
```

The nonnegative half of the Bogoliubov spectrum matches this to `1e-12` for `n = 2, …, 8` and for `n = 16, 32, 64`. The last three use only the `2n × 2n` matrix.

For `n ≤ 8`, the gap `spectrum[1] − spectrum[0]` of `transverse_ising(n, 1, 1)` equals `ε_1`.

Since `sin x < x`, `(2n + 1) ε_1 < 2π`. Assert that bound, and that `(2n + 1) ε_1` increases strictly over `n = 8, 16, 32, 64`. Do not hard-code a decimal for the limit.

## Translation

### 8. The translation operator

Add `translation(n)`, the permutation matrix with

```text
T |x_0 x_1 … x_{n−1}⟩ = |x_{n−1} x_0 … x_{n−2}⟩
```

Build it from bit operations, not from `pauli`. It is not `cyclic_shift`, which adds 1 to the basis index.

- `T` is unitary and `Tⁿ = I`.
- `T Z_i T† = Z_{i+1 mod n}` for every `i`, with `n = 3, 4, 5`.
- On three qubits, `T basis(4, 3) = basis(2, 3)`: `|100⟩ → |010⟩`.

### 9. Heisenberg ring

Add `heisenberg_ring(n)` for `n ≥ 3`:

```text
H = ∑_{i=0}^{n−1} (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1}),   indices mod n
```

For `n = 4, …, 8`, `H` commutes with `translation(n)`, with `total_spin(n)`, and with `∑ Z_i`, to `1e-12`. `basis(0, n)` has energy `n`. At `n = 2` the two bonds coincide, so reject it.

### 10. Momentum blocks

Add `momentum_blocks(H, n)`. For `k = 0, …, n − 1`, project onto the eigenspace of `T` with eigenvalue `e^{2πik/n}` using

```text
P_k = (1/n) ∑_{m=0}^{n−1} e^{−2πikm/n} T^m
```

and return each block's ascending spectrum.

For `heisenberg_ring(n)` with `n = 4, …, 8`, the block dimensions sum to `2ⁿ`, and the union of the block spectra equals the full spectrum to `1e-10`, multiplicities included.

### 11. One magnon

Restrict `heisenberg_ring(n)` to the `n` states with a single `1`. One flipped spin breaks two `ZZ` bonds, which gives `n − 4`. Since `XX + YY = 2(σ₊σ₋ + σ₋σ₊)`, the flip hops with amplitude 2, which gives `4 cos k`. So

```text
E(k) = n − 4 (1 − cos k),   k = 2πm/n
```

This matches to `1e-12` for `n = 4, …, 8`. The `k = 0` magnon is `S₋` applied to the ferromagnet, with `⟨S²⟩ = (n/2)(n/2 + 1)`. Every other magnon has `⟨S²⟩ = (n/2 − 1)(n/2)`: it lies in the same `S_z` sector and is orthogonal to that state.

### 12. Four sites from two sublattice spins

On four sites, `σ·σ = 4 S·S`, and the ring couples each even site to both odd sites:

```text
H = 4 S_A · S_B = 2 [S(S+1) − S_A(S_A+1) − S_B(S_B+1)]
```

with `S_A` the spin of sites `{0, 2}` and `S_B` the spin of sites `{1, 3}`. Count the levels:

- `(S_A, S_B, S) = (1, 1, 0)` gives `−8` once.
- `(1, 1, 1)` gives `−4`, three times.
- `(1, 1, 2)` gives `+4`, five times.
- `(1, 0, 1)`, `(0, 1, 1)` and `(0, 0, 0)` give `0`, seven times in all.

`spectrum(heisenberg_ring(4))` is exactly this list to `1e-12`.

### 13. The momentum of the ground state

For `n = 4, 6, 8` the ring's ground state is nondegenerate, has `⟨S²⟩ = 0`, and has `T` eigenvalue `(−1)^{n/2}`: `+1`, `−1`, `+1` (Marshall's sign rule). Assert the energy only at `n = 4`, where it is `−8`. No other ring energy is cited here.

## The toric code on a 2 × 2 torus

### 14. Stars and plaquettes

Eight edges, coordinates mod 2. Horizontal edge `h(x, y) = 2(2y + x)`, vertical edge `v(x, y) = 2(2y + x) + 1`.

- Star at `(x, y)`: `X` on `h(x, y), h(x−1, y), v(x, y), v(x, y−1)`.
- Plaquette at `(x, y)`: `Z` on `h(x, y), h(x, y+1), v(x, y), v(x+1, y)`.

Add `toric_code_stabilizers()`, returning stars and plaquettes in the order `(x, y) = (0,0), (0,1), (1,0), (1,1)`. Add `toric_code_hamiltonian()`, which is `−∑ A_s − ∑ B_p`. The labels are exactly:

```text
stars       XXXIIXII  IXIIXXXI  XIXXIIIX  IIIXXIXX
plaquettes  ZZIZZIII  ZIIIZZIZ  IZZZIIZI  IIZIIZZZ
```

All eight commute (`stabilizers_commute`). The product of the four stars is `I`, and so is the product of the four plaquettes. Three of each generate a group of `2⁶` elements (`stabilizer_group`), which leaves `8 − 6 = 2` logical qubits.

### 15. Spectrum by counting syndromes

An eigenstate violates `a` stars and `b` plaquettes, each count even, and has energy `−8 + 2(a + b)`. Each syndrome pattern carries the four-dimensional logical space, so the degeneracy of a level is `∑ 4 C(4, a) C(4, b)` over even `a, b` with that `a + b`:

```text
E   −8   −4   0     4    8
g    4   48   152   48   4
```

`spectrum(toric_code_hamiltonian())` reproduces this table exactly.

### 16. The ground space is the code space

`code_projector` on three stars and three plaquettes equals the projector onto the four `E = −8` eigenvectors to `1e-10`.

### 17. Logical loops

```text
Z̄₁ = Z on h(0,0), h(1,0)    ZIZIIIII
X̄₁ = X on h(0,0), h(0,1)    XIIIXIII
Z̄₂ = Z on v(0,0), v(0,1)    IZIIIZII
X̄₂ = X on v(0,0), v(1,0)    IXIXIIII
```

Each loop commutes with all eight stabilizers. `Z̄₁` anticommutes with `X̄₁`, and `Z̄₂` with `X̄₂`. Every other pair commutes. None is in the stabilizer group. `code_distance` on the six generators returns `2`: on a torus of side 2, a non-contractible loop has two edges.

## Circuits with an exact output

### 18. Teleportation

Qubit 0 holds `|ψ⟩`; qubits 1 and 2 hold `bell("phi+")`. Apply `CNOT(0→1)`, then `H` on qubit 0, and measure qubits 0 and 1. On outcome `(m₀, m₁)`, apply `Z^{m₀} X^{m₁}` to qubit 2.

For `|0⟩, |1⟩, |+⟩, |+i⟩` and `cos(0.3)|0⟩ + e^{0.7i} sin(0.3)|1⟩`, each outcome has probability `1/4`, and the corrected qubit 2 is `|ψ⟩` up to a global phase, to `1e-12`. Use `post_measure`, or projectors and `partial_trace`.

### 19. Teleportation through a Werner pair

Replace the Bell pair by

```text
ρ_W(p) = p |Φ⁺⟩⟨Φ⁺| + (1 − p) I/4
```

Averaged over outcomes, the output is `p ρ + (1 − p) I/2`. That is `depolarize(ρ, 3(1 − p)/4)`, and the fidelity with a pure input is `(1 + p)/2`. Check `p = 0.2, 0.5, 0.9` to `1e-12`.

The classical bound `2/3` is beaten exactly when `p > 1/3`, which is exactly when `concurrence(ρ_W(p)) = (3p − 1)/2` is positive. Assert both at `p = 0.3` and `p = 0.4`.

### 20. Superdense coding

Alice applies `Z^{b₀} X^{b₁}` to qubit 0 of `|Φ⁺⟩`. The four results are the four Bell states, mutually orthogonal. Bob applies `CNOT(0→1)`, then `H` on qubit 0, and reads `|b₀ b₁⟩` with probability 1. On `(1, 1)` the state is `−|11⟩`; the sign is a global phase.

### 21. Phase estimation

There are `m` counting qubits and one target, with `m + 1 ≤ 8`. Let `U = diag(1, e^{2πiφ})` and put the target in `|1⟩`. Counting qubit `k` (qubit 0 is the most significant) controls `U^{2^{m−1−k}}`. Before the inverse transform, the counting register is

```text
2^{−m/2} ∑_x e^{2πiφx} |x⟩
```

Apply `qft_matrix(m).conj().T`. With `δ = φ − j/2^m`, outcome `j` has probability

```text
P(j) = sin²(π 2^m δ) / (2^{2m} sin²(π δ)),   and P(j) = 1 when δ = 0
```

- `φ = 5/8` with `m = 3` returns `j = 5` with probability 1.
- `φ = 1/3` with `m = 4, 6` and `φ = 0.2137` with `m = 7` match the formula to `1e-12`.
- The most likely outcome has probability at least `4/π²`.

Build the controlled powers as matrices; the circuit has at most eight qubits.

## Information and metrology

### 22. Mutual information and conditional entropy

Add `mutual_information(rho, qubits, a, b)`, defined as `S_A + S_B − S_AB`, and `conditional_entropy(rho, qubits, a, b)`, defined as `S_AB − S_B`. Here `a` and `b` are disjoint qubit lists, every other qubit is traced out, and entropies are in bits via `von_neumann_entropy`.

- Bell pair: `I = 2` and `S(A|B) = −1`.
- `|0⟩ ⊗ |+⟩`: `I = 0`.
- `ghz(3)`, qubits 0 and 1: `I = 1` and `S(A|B) = 0`.
- `w_state(3)`, qubits 0 and 1: `ρ_AB` and `ρ_A` both have eigenvalues `2/3, 1/3`, so `I = h(1/3)` with `h` the binary entropy.

### 23. Holevo quantity

Add `holevo(probs, states)`, defined as `S(∑ p_i ρ_i) − ∑ p_i S(ρ_i)`.

- `|0⟩` and `|+⟩` with equal weight: the average state has eigenvalues `(1 ± 1/√2)/2`, so `χ = h((1 + 1/√2)/2)`.
- Two orthogonal states with equal weight: 1 bit.
- Two copies of the same state: 0.
- In every case, `χ ≤ H(p)`.

### 24. Quantum Fisher information of a pure state

Add `qfi(state, generator)`. For a ket it is `4 Var(G)`. Take `G = ∑_i Z_i / 2`:

- `|+⟩^{⊗n}`: `n`.
- `ghz(n)`: `n²`, because the two branches differ by `n` in `G`.
- `basis(0, n)`: `0`.

Check `n = 1, …, 8`.

### 25. Quantum Fisher information of a mixed state

For a density matrix with eigenpairs `(λ_i, |i⟩)`:

```text
F = 2 ∑_{λ_i + λ_j > 0} (λ_i − λ_j)² / (λ_i + λ_j) |⟨i|G|j⟩|²
```

On a pure state this equals `4 Var(G)` to `1e-10`. On one qubit with Bloch vector `r` and `G = Z/2`, `F = r_x² + r_y²`. Write that algebra in the test.

Dephase every qubit with the Kraus pair `√((1+λ)/2) I` and `√((1−λ)/2) Z`. On one qubit this reproduces `dephasing` with `λ = e^{−2γt}`. Then:

- Product `|+⟩^{⊗n}`: `F = n λ²`.
- `ghz(n)`: `F = n² λ^{2n}`. The state stays in the span of `|0…0⟩` and `|1…1⟩`, and its coherence shrinks by `λⁿ`.

GHZ beats the product state exactly when `n λ^{2n−2} > 1`. At `n = 4` it does for `λ = 0.95` and does not for `λ = 0.6`.

## A driven qubit

### 26. Midpoint evolution

Add `evolve_driven(hamiltonian_at, state, time, steps)`. Here `hamiltonian_at(t)` returns a `Hamiltonian` or a Hermitian matrix. Each step applies `exp(−i H(t_k + dt/2) dt)`, built with `eigensystem`. For a constant `H` it matches `evolve` to `1e-12`.

### 27. A rotating drive, solved exactly

```text
H(t) = −(ω₀/2) Z + (Ω/2)(cos ωt X − sin ωt Y)
```

With `R(t) = e^{iωtZ/2}`, `cos ωt X − sin ωt Y = R X R†`. In the frame `|ψ⟩ = R|φ⟩` the Hamiltonian is time-independent:

```text
H_eff = (δ/2) Z + (Ω/2) X,   δ = ω − ω₀
```

`R` is diagonal, so populations are frame-independent. From `|0⟩`:

```text
P₁(t) = Ω² / (Ω² + δ²) · sin²(√(Ω² + δ²) t / 2)
```

This is exact; no rotating-wave approximation is made. `evolve_driven` matches it to `1e-6` on resonance and at `δ = 0.1`, for `ω₀ = 1`, `Ω = 0.2`, `t = 7`. The formula is the test's oracle; do not use the frame inside `evolve_driven`.

The opposite sense of rotation, `cos ωt X + sin ωt Y = R† X R`, goes through the frame `R†` instead. Its detuning is `ω + ω₀`, so the same formula holds with `δ → ω + ω₀`. At `ω = ω₀`, `P₁` never exceeds `Ω²/(Ω² + 4ω₀²)`. Check that formula at `t = 7` to `1e-6`.

### 28. Second order

At `ω₀ = 1`, `Ω = 0.4`, `ω = 1.2`, `t = 10`, the error in `P₁` falls by a factor in `[3.9, 4.1]` each time the steps double from 100 to 800.

## Open systems

### 29. Matrix exponential

Add `expm(A)` for any square complex matrix, by scaling and squaring a truncated Taylor series. Do not use `eig`: a Liouvillian need not be diagonalizable.

- For a Hermitian `A`, it matches the `eigh` exponential to `1e-12`.
- `expm([[0, t], [0, 0]]) = [[1, t], [0, 1]]` exactly, for a matrix with no eigenbasis.
- `expm(−iθX/2) = cos(θ/2) I − i sin(θ/2) X`.
- `expm(A) expm(−A) = I` for a fixed random `A`.

### 30. The Lindbladian

Use row-major vectorization, `vec(ρ) = ρ.reshape(−1)`. Then `vec(AρB) = (A ⊗ Bᵀ) vec(ρ)`, and

```text
L = −i (H ⊗ I − I ⊗ Hᵀ) + ∑_k [ L_k ⊗ L_k* − ½ (L_k†L_k ⊗ I) − ½ (I ⊗ (L_k†L_k)ᵀ) ]
```

Add `lindbladian(H, jumps)`. Add `lindblad_evolve(H, jumps, rho, time)`, which returns the reshaped `expm(L t) vec(ρ)`.

- `vec(I)ᵀ L = 0`: the trace is preserved.
- With no jumps, the result is `U ρ U†`.
- The output is Hermitian, has trace 1, and has eigenvalues at least `−1e-12`.

### 31. The closed forms, recovered

- Jump `√γ σ₋` reproduces `amplitude_damping_evolve`, and equals `amplitude_damping(ρ, 1 − e^{−γt})`.
- Jump `√γ Z` gives `γ(ZρZ − ρ)` and reproduces `dephasing`.

Check `|1⟩`, `|+⟩` and `|+i⟩` at several times to `1e-10`.

### 32. A driven, damped qubit

Add `lindblad_steady_state(H, jumps)`. Solve `L vec(ρ) = 0` with one equation replaced by `Tr ρ = 1`.

With `H = (Ω/2) X` and jump `√γ σ₋`, the excited population is

```text
ρ₁₁ = Ω² / (γ² + 2Ω²)
```

which never exceeds `1/2`. Check `(γ, Ω) = (0.7, 1.3)` and two more pairs to `1e-10`. `lindblad_evolve` at `γt = 50` lands on the same state to `1e-8`.

### 33. Detailed balance gives Gibbs

With `H = −(ω/2) Z`, `n̄ = 1/(e^{βω} − 1)`, and jumps `√(γ(n̄+1)) σ₋` and `√(γ n̄) σ₊`, the steady state has `ρ₁₁/ρ₀₀ = n̄/(n̄+1) = e^{−βω}`. It equals `gibbs(H, β)` to `1e-10`.

Do the same for two qubits, each with its own bath, under `H = −(ω/2)(ZI + IZ)`.

### 34. Entanglement sudden death

Send `α|00⟩ + β|11⟩` through `amplitude_damping(p)` on each qubit. The result is an X state with

```text
ρ_{11,11} = |β|²(1−p)²,  ρ_{01,01} = ρ_{10,10} = |β|² p(1−p),  ρ_{00,11} = αβ*(1−p)
```

so

```text
C = 2(1 − p) max(0, |αβ| − p|β|²)
```

- Equal amplitudes give `C = (1 − p)²`, which never vanishes.
- `|α|² = 0.2`, `|β|² = 0.8` gives `C = 0` from `p = |α|/|β| = 1/2` onward, so at `t = ln 2 / γ` in continuous time.

Match `concurrence` to `1e-12` at `p = 0.1, 0.3, 0.6`. Match `lindblad_evolve` with jumps `√γ σ₋ ⊗ I` and `√γ I ⊗ σ₋` at the same `p`.

## A qubit and an oscillator

Put this in a new module, `src/eigenlab/oscillator.py`. The oscillator is truncated at `d` levels. In a product, the oscillator is the left factor.

### 35. Truncated ladder

Add `annihilation(d)`, with `a|n⟩ = √n |n−1⟩`, and `number(d) = a†a = diag(0, …, d−1)`.

`[a, a†]` is `I` except its last diagonal entry, which is `1 − d`. That is the truncation, written in the test. `a†a + 1/2` has the spectrum `n + 1/2` exactly.

### 36. Coherent states

Add `coherent(alpha, d)` with amplitudes `c_n = e^{−|α|²/2} αⁿ/√(n!)` for `n < d`, unnormalized.

- `‖c‖²` is the Poisson sum `∑_{n<d} e^{−|α|²} |α|^{2n}/n!`, to `1e-12`.
- `a c = α c − α c_{d−1} |d−1⟩` exactly: only the top level breaks the eigenvalue equation.
- At `α = 2`, `d = 30`, `1 − ‖c‖²` is below `1e-12`.

### 37. Jaynes–Cummings

```text
H = ω_c a†a ⊗ I − (ω_q/2) I ⊗ Z + g (a† ⊗ σ₋ + a ⊗ σ₊)
```

It commutes with `a†a ⊗ I + I ⊗ |1⟩⟨1|`.

- The ground state is `|0⟩|0⟩`, at `−ω_q/2`.
- The block `{|n⟩|1⟩, |n+1⟩|0⟩}` has, with `δ = ω_q − ω_c`,

  ```text
  E_{n,±} = (n + ½) ω_c ± ½ √(δ² + 4g²(n+1))
  ```

- `|d−1⟩|1⟩` has no partner after truncation and sits alone at `(d−1) ω_c + ω_q/2`.

The full spectrum is exactly these `2d` values to `1e-12`, for `d = 12`, `ω_c = 1`, `ω_q = 1.15`, `g = 0.05`.

### 38. Vacuum Rabi oscillation

From `|0⟩|1⟩`:

```text
P_e(t) = 1 − (4g² / (δ² + 4g²)) sin²(√(δ² + 4g²) t / 2)
```

which is `cos²(gt)` on resonance.

From a coherent field times `|1⟩`, on resonance, `P_e(t) = ∑_n p_n cos²(g √(n+1) t)`. Here `p_n` is the normalized coherent weight, with the field supported on `n ≤ d − 2` so that every block it touches is complete. `evolve` on the Jaynes–Cummings matrix matches both to `1e-10`. Do not hard-code a collapse or revival time.

## The scope line

### 39. What this haul added

When phases 1–38 are green, update `docs/scope.md` and the README so these are described as present:

- the parity encoding and single-qubit `Z` tapering, with H₂ on two qubits and HeH⁺ on three
- the exact modes of the critical Ising chain
- translation symmetry, momentum blocks, the Heisenberg ring, and its one-magnon band
- the toric code on a 2 × 2 torus
- teleportation, superdense coding, and phase estimation
- mutual information, conditional entropy, the Holevo quantity, and quantum Fisher information
- time-dependent evolution with the exact rotating-drive solution
- the Lindblad equation, its steady states, detailed balance, and entanglement sudden death
- a truncated oscillator, coherent states, and the Jaynes–Cummings model

Leave these named as not done:

- Any molecule beyond H₂, HeH⁺, and H₄.
- Any basis beyond STO-3G s functions.
- Exact diagonalization above eight qubits.

## Done

After phase 39 the lab is still a state, a Hamiltonian, and a spectrum. It now has three encodings and drops qubits a symmetry fixes. A ring has a momentum, a torus has a code, a qubit has a bath, and a qubit has an oscillator. No new dependency. The eight-qubit diagonalization limit is where it was.
