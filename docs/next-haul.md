# Next haul

The first haul is in the tree. Jordan–Wigner, the spin models, and STO-3G H₂ are already tested. This haul adds the encoding that was left out, the free-fermion solution of a model already in `hamiltonian.py`, the spin and orbital content of states the eigensolver already returns, one derivative, and one second molecule.

Eight phases, in order. Each one is a single change set. Do not start phase k+1 until phase k is green under `pytest`.

## Contract

Stay inside this stack: Python 3.11, NumPy, pytest. Do not add Qiskit, PennyLane, OpenFermion, or PySCF.

`pauli("XZ")` means qubit 0 is the leftmost character and the most significant bit of `basis(index, qubits)`. Orbital 0 is that same qubit. `spatial_to_spin_orbital` interleaves spins as `(↑, ↓, ↑, ↓, …)`. Do not flip either convention.

Jordan–Wigner remains the default. Existing functions keep their current return values. Bravyi–Kitaev is a second set of operators, not a replacement.

Build matrices only for n ≤ 8. A test number is a few lines of algebra written next to the assertion, or a value copied from a citation named in the test module.

Do not bump the package version.

Out of this haul: shot noise, variational algorithms, a third molecule, systems above eight qubits, and a Hellmann–Feynman force on the tabulated H₂ curve. Those four bond lengths are a table, not a differentiable function.

## Encoding

### 1. Bravyi–Kitaev operators

Add `bravyi_kitaev_a` and `bravyi_kitaev_adag`. They return a `Hamiltonian` term list.

The occupation string `x` and the Bravyi–Kitaev string `β` are related by the Fenwick partial sums, over GF(2). Number qubits from 0. One-based index `i = q + 1` stores the sum of occupations whose zero-based indices run from `i - (i & -i)` through `i - 1` inclusive. Then `β_0 = x_0`, `β_1 = x_0 + x_1`, `β_2 = x_2`, `β_3 = x_0 + x_1 + x_2 + x_3`, and so on. The matrix `M` of this map is unit triangular, hence invertible. The empty string is a fixed point, so the vacuum is still `basis(0, n)`.

Let `Π` be the permutation of computational basis states that sends `|x⟩` to `|β⟩`. Then

```text
a_p^{BK} = Π a_p^{JW} Π^T
```

with `Π^{-1} = Π^T`. Implement this by conjugating the two Jordan–Wigner Majorana strings through a CNOT circuit for `M`, using the Pauli update rules for CNOT. Do not build `Π` as a matrix for the operator that ships in the library. Do use that matrix, for `n ≤ 4`, as the test oracle: `bravyi_kitaev_a(p, n).to_matrix()` equals `Π @ jordan_wigner_a(p, n).to_matrix() @ Π.T` for every orbital.

For every `n ≤ 5` the matrices satisfy `{a_i, a†_j} = δ_ij I`, `{a_i, a_j} = 0`, and `a_i² = 0`. The dagger of `bravyi_kitaev_a` is `bravyi_kitaev_adag`. For `n ≥ 2` the Pauli strings differ from Jordan–Wigner; returning the Jordan–Wigner operators is not this phase.

`a†_p` applied to the vacuum has `⟨N⟩ = 1`.

### 2. Same spectra, different strings

Rebuild three Hamiltonians with the Bravyi–Kitaev operators, by the same algebraic formulas already used with Jordan–Wigner.

- Open tight-binding chain at `n = 3, 4, 5`. Eigenvalues match `tight_binding_chain` to `1e-8`.
- Hubbard dimer at the `(t, U)` pairs already covered by `test_hubbard_dimer`. The full spectra match, including degeneracies.
- H₂ STO-3G at `R = 1.401` a.u., through `integral_hamiltonian` and the same integral arrays `h2_sto3g_integrals` already returns. The ground-state energy matches the Jordan–Wigner Hamiltonian to `1e-8`.

Add a mapping argument where a builder needs one. The default stays Jordan–Wigner, and the current tests keep passing without being rewritten around Bravyi–Kitaev.

## The Ising chain as free fermions

### 3. Bogoliubov spectrum of the open transverse Ising chain

`transverse_ising` is

```text
H = −J ∑_{i=0}^{n-2} Z_i Z_{i+1} − h ∑_i X_i
```

A Hadamard on every qubit leaves the spectrum unchanged and turns this into

```text
H' = −J ∑_{i=0}^{n-2} X_i X_{i+1} − h ∑_i Z_i
```

Under this repo's Jordan–Wigner convention,

```text
Z_i = 1 − 2 a†_i a_i
X_i X_{i+1} = (a†_i − a_i)(a_{i+1} + a†_{i+1})
            = (a†_i a_{i+1} + h.c.) + (a†_i a†_{i+1} + h.c.)
```

So `H'` is quadratic:

```text
C = −n h
A_ii = 2h
A_{i,i+1} = A_{i+1,i} = −J
B_{i,i+1} = −J
B_{i+1,i} = +J
```

with `C` the additive constant, `A` the normal matrix, and `B` the pairing matrix (`B` antisymmetric). Diagonalize the `2n × 2n` Bogoliubov–de Gennes matrix

```text
[[ A,  B ],
 [ −B*, −A* ]]
```

Its eigenvalues come in `±ε_k` pairs. Take the `n` algebraically largest; they are nonnegative, and a zero mode may sit on the cut. Every many-body energy is a choice of occupations `ν_k ∈ {0, 1}`:

```text
E(ν) = ∑_k (ν_k − 1/2) ε_k
```

`C + (1/2) Tr(A) = 0`, which is why no extra shift appears.

The test builds this Bogoliubov matrix itself. It does not call `spectrum` on the spin operator and rename the result. For `n = 2, 3, 4, 5`, and for `(J, h)` in `{(1, 0), (1, 0.5), (1, 1), (0.7, 1.3)}`, the `2^n` values of `E(ν)` match `spectrum(transverse_ising(n, J, h))` to `1e-8`, degeneracies included.

One-site sanity check, written next to the assertion: `H = −h Z` has energies `−h` and `+h`, and the formula with `A = [2h]`, `B = 0`, `ε = 2h` reproduces them.

## What a ground state is made of

### 4. Total spin

Add `total_spin` for qubits and `total_spin_squared` for the interleaved fermion ordering already used by `total_spin_z`.

For qubits, `S_z^{(i)} = Z_i / 2` and `S_+^{(i)} = (X_i + i Y_i) / 2`. Then

```text
S² = S_- S_+ + S_z² + S_z
```

On two qubits the singlet `(|01⟩ − |10⟩) / √2` has `⟨S²⟩ = 0`. Each of the three triplet states has `⟨S²⟩ = 2`. The same numbers follow from `S² = 3/2 + (XX + YY + ZZ) / 2` and the XXZ energies already tested: the singlet of `XX + YY + ZZ` sits at `−3`, the triplet at `+1`.

For fermions, `S_+ = ∑_i a†_{2i} a_{2i+1}` and `S_- = S_+†`, with `a` the Jordan–Wigner operators. `S_z` is `total_spin_z`. The same quadratic formula gives `S²`.

On the Hubbard dimer at `t = 1`:

- The lowest `N = 2`, `S_z = 0` state at `U = 0` and at `U = 4` has `⟨S²⟩ = 0`. Its energy is the half-filled singlet `(U − √(U² + 16 t²)) / 2`. At large `U` this is not the ground state of the full space; tag the sector before measuring.
- The `N = 2`, `S_z = 0` triplet has energy `0` and `⟨S²⟩ = 2` at those same parameters. Singly occupied triplets do not feel `U` or the hopping.

The H₂ ground state at `R = 1.401` a.u. has `⟨S²⟩ = 0`.

### 5. One-particle density matrix

Add `one_particle_rdm(state, n)` returning `γ_{pq} = ⟨a†_p a_q⟩`, and `natural_occupations` as the eigenvalues of `γ` sorted descending.

Use Jordan–Wigner operators and states in the occupation basis.

- `γ` is Hermitian.
- `Tr(γ) = ⟨N⟩`.
- Every natural occupation lies in `[0, 1]`.
- The computational state `a†_i a†_j |0⟩` has natural occupations `1, 1, 0, …` with the two `1`s on orbitals `i` and `j`.
- The Hubbard dimer at `U = 0`, `t = 1`, full-space ground state (both electrons in the bonding orbital) has natural occupations `1, 1, 0, 0`.

On the Jordan–Wigner H₂ ground state the occupations come in two equal pairs and sum to 2. At `R = 1.401` a.u. the largest occupation is above `0.95`. At `R = 3.0` a.u. it is strictly smaller, and still above `1/2`. Stretching the bond moves the state off a closed-shell determinant. Do not hard-code a finer decimal.

## One derivative

### 6. Hellmann–Feynman

For a normalized eigenstate, `dE/dλ = ⟨∂H/∂λ⟩`. Three parameters already enter the Hamiltonians analytically.

- XXZ dimer, `H = XX + YY + Δ ZZ`, so `∂H/∂Δ = ZZ`. On the singlet `⟨ZZ⟩ = −1`, and the energy `−2 − Δ` has the same slope. On `|00⟩`, `⟨ZZ⟩ = 1` and the energy is `Δ`.
- Open transverse Ising, `n = 3`, `J = 1`, `h = 1`. `∂H/∂h = −∑_i X_i`. A central difference of the ground energy with step `10^{-4}` matches that expectation to `10^{-6}`.
- Hubbard dimer, `t = 1`, on the half-filled singlet rather than the full-space ground state. `E(U) = (U − √(U² + 16 t²)) / 2`, so

```text
dE/dU = (1 − U / √(U² + 16 t²)) / 2
```

and this equals `⟨n_{0↑} n_{0↓} + n_{1↑} n_{1↓}⟩`. Check `U = 0`, where the derivative is `1/2`, and `U = 4`, from the algebraic expression.

Do not difference the four tabulated H₂ energies and report a bond force. The integral table has no nearby pair of geometries from which `∂H/∂R` can be built.

## The second molecule

### 7. HeH⁺ in STO-3G

One geometry. Transcribe the STO-3G integrals from a single named reference that prints enough of the chemist-notation tensors to fill `integral_hamiltonian`, and that prints a full-CI energy for that same geometry and basis. Record the bond length, the unit, the nuclear repulsion `2/R`, and whether the printed energy already includes repulsion. Match the printed precision the way the H₂ test does. If two references disagree at the precision they both claim, stop this phase and write the disagreement into the test docstring.

HeH⁺ has two electrons. The ground state from this Hamiltonian has `N = 2`, `S_z = 0`, and `⟨S²⟩ = 0`. Build it with Jordan–Wigner, then again with Bravyi–Kitaev; the two ground energies agree to `1e-8`.

Do not add a second bond length. Do not add LiH, BeH₂, or H₃⁺.

### 8. The scope line

When the seven phases above are green, update `docs/scope.md` and the README table so Bravyi–Kitaev, the free-fermion Ising spectrum, `S²`, natural occupations, the three Hellmann–Feynman checks, and HeH⁺ are described as present.

Leave these named as not done: a derivative of the H₂ curve, any molecule beyond H₂ and HeH⁺, and exact diagonalization above eight qubits.

## Done

After phase 8 the lab is still a state, a Hamiltonian, and a spectrum. The same H₂ number now comes out of two encodings. The transverse Ising spectrum comes out of a `2n × 2n` matrix as well as out of `2^n`. The ground states carry a spin and a set of occupations. One molecule was added, from a book, on the eigensolver that was already here.
