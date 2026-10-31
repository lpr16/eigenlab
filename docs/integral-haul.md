# Integral haul

The first three hauls are in the tree. The molecules are still a lookup table: `h2_sto3g_integrals` knows four bond lengths and `heh_sto3g_integrals` knows one. This haul computes those integrals from s-type Gaussians, so the H₂ curve becomes a function with a derivative, and a third molecule fits inside eight qubits. It then adds three tracks that also fit there exactly: quantum error correction, thermal states, and free-fermion entanglement. Thirty-eight phases, in order. Each one is a single change set with a closed-form or cited check.

Do not start phase k+1 until phase k is green under `pytest`.

This file is the spec for an agent. Point the agent at one phase. If you are asked to do the whole haul, finish phase k, leave `pytest` green, then start phase k+1. Do not implement two phases behind one test run. Do not bump the package version. Completion of a phase is the new assertions passing, not a summary of intent.

## Contract

Stay inside this stack: Python 3.11, NumPy, pytest, and the standard library. `math.erf` is how the Boys function is evaluated. Do not add SciPy, PySCF, OpenFermion, Qiskit, PennyLane, or any integral package.

`pauli("XZ")` means qubit 0 is the leftmost character and the most significant bit of `basis(index, qubits)`. Orbital 0 is that same qubit. `spatial_to_spin_orbital` interleaves spins as `(↑, ↓, ↑, ↓, …)`. Jordan–Wigner stays the default. Do not flip any of these.

Lengths are in bohr, energies in hartree. A primitive s Gaussian centred at `A` is `(2α/π)^{3/4} exp(−α |r − A|²)`, normalized.

Build matrices only for n ≤ 8. Do not diagonalize anything above eight qubits, and do not add a matrix-free solver to get around that.

A test number is one of three things: a few lines of algebra written next to the assertion, a value this repo already cites, or a value from a citation named in the test module with its table or page. The values already cited here are the H₂ table in `h2_sto3g_integrals`, the HeH⁺ atomic-orbital integrals and orbital coefficients in the `heh_sto3g_integrals` docstring, `E_RHF = −2.860662` in `test_heh_sto3g`, and `H_ATOM_STO3G_ENERGY`. When a phase asks for a cited value and no source is at hand, its internal checks stand alone and the test says so in a comment. Do not invent a decimal to make a test pass. Do not loosen a tolerance until a fit succeeds.

Six-decimal transcribed integrals are compared to `1e-5`. If a computed value disagrees with a transcribed one beyond that, stop and report which entry and by how much. Do not edit the transcribed value, and do not widen the tolerance.

Out of this haul: p orbitals and any basis beyond STO-3G s functions, a fourth molecule, open-shell or unrestricted Hartree–Fock, variational algorithms, shot noise, hardware, and exact diagonalization above eight qubits.

## Integrals over s Gaussians

Put this in a new module, `src/eigenlab/integrals.py`. It imports nothing from `molecular.py`.

### 1. Overlap

Add `gaussian_overlap(a, A, b, B)` for two normalized primitives with exponents `a`, `b` and centres `A`, `B`.

With `p = a + b` and `μ = ab / p`, the Gaussian product theorem gives

```text
S = (4ab / p²)^{3/4} exp(−μ |A − B|²)
```

Write that derivation in the test. Self-overlap is 1 for every `a`. `S` is symmetric in its two arguments, equals 1 only when `a = b` and `A = B`, and decreases as `|A − B|` grows.

### 2. STO-3G contraction

Add `sto3g_1s(zeta, center)` for an STO-3G 1s function. For `ζ = 1` the exponents are `(0.109818, 0.405771, 2.22766)` and the contraction coefficients are `(0.444635, 0.535328, 0.154329)`, from Hehre, Stewart and Pople, J. Chem. Phys. 51, 2657 (1969). Exponents scale as `ζ²`; coefficients do not.

The contracted self-overlap is 1 to `1e-5`, since the coefficients are printed to six digits.

At `R = 1.4632` with `ζ_He = 2.0925` and `ζ_H = 1.24`, the contracted overlap matches the `S_12 = 0.450770` already transcribed in the `heh_sto3g_integrals` docstring.

### 3. Kinetic energy

Add the kinetic integral for two primitives,

```text
T = μ (3 − 2μ |A − B|²) S
```

and its contraction. For one normalized primitive at one centre, `T = 3a/2`. Write the algebra in the test.

### 4. Boys function and nuclear attraction

Add `boys0(t) = ½ √(π/t) erf(√t)`, with `boys0(0) = 1`. For small `t`, the series `1 − t/3 + t²/10` matches it to `1e-10` at `t = 1e-3`.

Add the nuclear-attraction integral of a charge `Z` at `C` between two primitives,

```text
V = −Z (2π / p) K F0(p |P − C|²) × (normalizations),   P = (aA + bB)/p,   K = exp(−μ |A − B|²)
```

and its contraction. For one normalized primitive with the nucleus at its centre, `V = −2Z √(2a/π)`. Write the algebra in the test.

The STO-3G hydrogen atom, `ζ = 1.24`, has `T + V` equal to `H_ATOM_STO3G_ENERGY` to `1e-5`.

The HeH⁺ one-electron integrals `h_11`, `h_12`, `h_22` match the docstring of `heh_sto3g_integrals`.

### 5. Electron repulsion

Add the two-electron integral `(ab|cd)` in chemist notation for four primitives,

```text
(ab|cd) = 2π^{5/2} / (p q √(p + q)) K_ab K_cd F0(pq/(p+q) |P − Q|²) × (normalizations)
```

and its contraction. For four copies of one normalized primitive at one centre, `(aa|aa) = 2√(a/π)`. Write the algebra in the test.

The integrals have the real eightfold symmetry. The six HeH⁺ values `(11|11)`, `(21|11)`, `(21|21)`, `(22|11)`, `(22|21)`, `(22|22)` match the docstring of `heh_sto3g_integrals`.

### 6. The AO bundle

Add `ao_integrals(atoms)`, where `atoms` is a list of `(Z, zeta, position)`. It returns `S`, the core Hamiltonian `h = T + V`, the electron-repulsion array, and `V_nuc = Σ_{i<j} Z_i Z_j / |R_i − R_j|`.

The arrays are symmetric as they should be. For HeH⁺ at `R = 1.4632`, every transcribed number from phases 2–5 comes out of this one call.

## H₂ from integrals

### 7. Minimal-basis orbitals

For two hydrogen 1s functions `φ_1`, `φ_2` with overlap `S`, symmetry alone fixes the orbitals:

```text
σ_g = (φ_1 + φ_2) / √(2(1 + S)),   σ_u = (φ_1 − φ_2) / √(2(1 − S))
```

Add `mo_transform(C, h, eri)`, returning MO integrals in chemist notation. With `C = [σ_g, σ_u]`, `Cᵀ S C = I` to `1e-12`.

### 8. The table, reproduced

At `R = 1.0, 1.401, 2.0, 3.0`, the MO integrals from phase 7 match the `h2_sto3g_integrals` table to `1e-5`, entry by entry. The full-CI energies of the existing H₂ tests then follow unchanged.

The table is labelled `R = 1.401`, while Szabo & Ostlund print `R = 1.4`. Compare at the label the table carries. If an entry disagrees, stop and report it under the rule in the contract.

### 9. Any bond length

`h2_sto3g_integrals(R)` computes the integrals for any `R > 0`. The four tabulated geometries move into a private table that only tests read, as the oracle for phase 8.

The existing H₂ tests stay green without being rewritten. `R ≤ 0` raises `ValueError`.

## Hartree–Fock

### 10. Symmetric orthogonalization

Add `lowdin(S)`, returning `X = S^{−1/2}` from `eigh`. `Xᵀ S X = I` to `1e-12`. Raise `ValueError` if `S` has an eigenvalue below `1e-10`.

### 11. Restricted Hartree–Fock

Add `rhf(ao, n_electrons)` for an even electron count. Start from the core-Hamiltonian guess. Solve `F C = S C ε` through `lowdin`, with density `P = 2 C_occ C_occᵀ`. Stop when the energy changes by less than `1e-10` and the density by less than `1e-8`. Density damping is allowed. If the iteration cap is reached, raise. Return the total energy, orbital energies, coefficients, and density.

For HeH⁺ at `R = 1.4632`, the energy is `−2.860662` to `1e-5`. That value is already cited in `test_heh_sto3g`. The coefficients match the docstring's `C` to `1e-5`, each column up to its sign.

### 12. HeH⁺ from integrals

`heh_sto3g_integrals(R)` computes its MO integrals through `rhf` for any `R > 0`. At `R = 1.4632` they match the existing table to `1e-5`, up to orbital signs. Fix the sign of each column so its largest coefficient is positive, and say so in the docstring.

The existing HeH⁺ test asserts that an unsupported bond length raises `ValueError`. That is the one assertion this haul rewrites: it now asserts that `R ≤ 0` raises.

### 13. H₂ RHF is the symmetry orbitals

For H₂ at `R = 1.0, 1.401, 2.0, 3.0`, `rhf` returns `σ_g` and `σ_u` up to sign, to `1e-8`.

The RHF energy equals `⟨Φ|H|Φ⟩` for the qubit Hamiltonian, where `Φ` is the basis state with spin-orbitals 0 and 1 filled, `basis(0b1100, 4)`. Full CI is below RHF at every one of those geometries.

Full CI does not depend on which orthonormal orbitals span the space. The ground energy from Löwdin orbitals `X` equals the one from `σ_g`, `σ_u` to `1e-10`.

## The curve as a function

### 14. Dissociation

On `R` from `0.8` to `6.0` in steps of `0.1`, plus `R = 10`, the full-CI energy is finite and has a single minimum on the grid.

At `R = 10`, full CI is `2 × H_ATOM_STO3G_ENERGY` to `1e-5`. The ground state's four spin-orbital natural occupations are each `1/2` to `1e-3`, because the state is `(σ_g² − σ_u²)/√2`.

RHF does not dissociate correctly: the RHF minus full-CI gap at `R = 6` is larger than at `R = 1.401`.

### 15. One derivative

Add `h2_energy_gradient(R, step=1e-4)`, a central difference of the full-CI ground energy.

The derivative is negative at `R = 1.0` and positive at `R = 2.0` and `R = 3.0`.

Matrix Hellmann–Feynman holds, because `H(R)` is a Hermitian matrix that depends smoothly on `R`. `⟨ψ| (H(R+h) − H(R−h)) / 2h |ψ⟩` on the ground state `ψ` matches the gradient to `1e-6` at `R = 1.0, 1.401, 2.0`.

### 16. The Pulay term

The electrostatic Hellmann–Feynman force moves the nuclei, while every basis function stays at its original centre. It is the derivative of `V_ne + V_nn` with the basis frozen, evaluated on the full-CI ground state.

At `R = 1.401` this force differs from the phase-15 gradient by more than `1e-3`. The difference is the Pulay term, which comes from basis functions that ride with the nuclei. The test prints both numbers. If they agree, stop and report: either the basis is not moving, or one derivative is wrong.

### 17. Equilibrium bond length

Bisect `h2_energy_gradient` on `[1.0, 2.0]` until the interval is shorter than `1e-6`. The sign change comes from phase 15. Call the result `R_e`.

`E(R_e) ≤ E(1.401)`, and both are within `1e-3` of the cited `−1.13727`.

Do the same with the RHF energy. `R_e(RHF) < R_e(FCI)`: correlation lengthens the bond.

If a cited STO-3G full-CI `R_e` is at hand, compare it, and name the table in the test.

### 18. Harmonic frequency

`k = d²E/dR²` at `R_e`, by central second difference. Steps `1e-2` and `5e-3` agree to `1e-4` relative, and `k > 0`.

`ω = √(k/μ)` with `μ = m_H / 2`. Take the masses and the hartree-to-cm⁻¹ conversion from CODATA 2018, and name them in the test.

If a cited STO-3G full-CI frequency is at hand, compare it, and name the table in the test.

## A third molecule: H₄

H₄ in STO-3G has four spatial orbitals, eight spin-orbitals, and eight qubits.

### 19. Rectangle integrals

Add `h4_sto3g_integrals(a, b)` for four hydrogens at `(±a/2, ±b/2, 0)` with `ζ = 1.24`. Use Löwdin orbitals by default, because full CI does not depend on that choice. Also add `h4_sto3g_hamiltonian(a, b, mapping="jordan_wigner")`, built through `integral_hamiltonian`.

The Hamiltonian has eight qubits and is Hermitian. It commutes with `total_number_operator(8)` and with `total_spin_z(4)` to `1e-10`.

Work in the `N = 4` block. `occupation_block` gives dimension `C(8, 4) = 70`.

### 20. Orbitals do not move full CI

At `a = 1.401`, `b = 2.0`, the `N = 4` ground energy from Löwdin orbitals equals the one from `rhf` orbitals to `1e-8`.

The full Jordan–Wigner and Bravyi–Kitaev spectra match to `1e-8`.

### 21. Two far molecules

Set `a = 1.401` for the bond and `b = 20` for the separation.

Full CI in `N = 4` is twice the phase-8 H₂ energy at `R = 1.401`, to `1e-6`. Both RHF and full CI are size-consistent for two closed-shell fragments, so RHF is also twice the H₂ RHF energy, to `1e-6`.

### 22. Four far atoms

At `a = b = 20`, full CI in `N = 4` is `4 × H_ATOM_STO3G_ENERGY` to `1e-5`. The lowest level is degenerate, from the spin states of four independent electrons. Do not assert a single ground vector there.

### 23. The square

Full CI is symmetric under swapping the sides: `E(a, b) = E(b, a)` to `1e-10`.

At `a = b = 2.0`, the derivative of `E(a + δ, a − δ)` at `δ = 0` vanishes by symmetry: a central difference is below `1e-6`.

In Löwdin orbitals, the one-electron Hamiltonian `h` has a doubly degenerate middle pair at the square, to `1e-8`, and that pair splits when `a ≠ b`. Four s orbitals on a square span `a_1g ⊕ e_u ⊕ b_2g`, and `e_u` is the pair.

The RHF minus full-CI gap is larger at the square `a = b = 2.0` than at the rectangle `a = 1.401`, `b = 2.0`. That inequality is a physics claim, not a target. If it fails, stop and report. Do not move the geometry until it passes.

### 24. What the H₄ ground state is

At the rectangle `a = 1.401`, `b = 2.0`, and at the square `a = b = 2.0`, embed the `N = 4` ground vector back into the 256-dimensional space.

`tag_sectors` gives `(4, 0)`, and `⟨S²⟩ = 0` to `1e-8` through `total_spin_squared(4)`. The natural occupations sum to 4. At the square, more of them lie strictly between `0.1` and `0.9` than at the rectangle.

## Error correction

Put this in a new module, `src/eigenlab/codes.py`. Stabilizers are Pauli labels, and their algebra goes through `pauli_mult`, `pauli_comm`, and `apply_pauli`.

### 25. Independent bit flips

Add `bit_flip_channel(rho, p)` for `n ≤ 8` qubits. Each qubit independently gets Kraus operators `√(1−p) I` and `√p X`. The trace is preserved, and on one qubit the map is `(1−p) ρ + p XρX`.

### 26. The bit-flip code

The stabilizers `ZZI` and `IZZ` commute, and `Π (I + S)/2` has rank 2. The four syndromes distinguish `III`, `XII`, `IXI`, and `IIX`.

Encode `α|000⟩ + β|111⟩`, apply `bit_flip_channel`, measure the syndrome, and correct by majority. The logical failure probability is

```text
3p²(1−p) + p³ = 3p² − 2p³
```

Check `p = 0, 0.1, 0.5`. Below `p = 1/2` this is less than `p`, and at `p = 1/2` both equal `1/2`.

### 27. The phase-flip code

Conjugate the bit-flip code by `H` on every qubit, through `conjugate_clifford`. The result is the stabilizers `XXI` and `IXX`. That code corrects single `Z` errors with the same `3p² − 2p³`.

The bit-flip code does not correct them. `ZII` commutes with `ZZI` and `IZZ`, is not in their group, and flips the logical phase.

### 28. The five-qubit code

The generators are `XZZXI`, `IXZZX`, `XIXZZ`, and `ZXIXZ`. They commute pairwise, and the code space has rank 2.

The identity and the 15 single-qubit Paulis give 16 distinct syndromes. That is all `2⁴` of them, which is why the code is called perfect.

### 29. Distance three

No Pauli of weight 1 or 2 commutes with all four generators. There are `15 + 90 = 105` such labels; check each one.

`XXXXX` and `ZZZZZ` commute with every generator, are not in the stabilizer group, and anticommute with each other. Some weight-3 Pauli also commutes with every generator without being in the group. Find one and assert it.

### 30. Knill–Laflamme

Take `P` as the code projector and `E` ranging over the identity and the 15 single-qubit Paulis. For the five-qubit code, `P E_a† E_b P = c_ab P` for all 256 pairs, with `c_ab` read off as `Tr(P E_a† E_b P) / 2`.

For the bit-flip code with `E` over `I` and the single-qubit `X` and `Z`, the condition fails, and the test names one failing pair.

### 31. The Steane code

Use seven qubits. Take the X-type and Z-type generators from the rows `0001111`, `0110011`, `1010101` of the Hamming `[7, 4]` parity-check matrix.

The six generators commute, and the code space has rank 2. Transversal `H` maps the stabilizer group to itself, swapping X-type and Z-type generators. It acts on the code as logical `H`: `H^{⊗7}` sends `|0_L⟩` to `|+_L⟩`.

## Thermal states

Put this in a new module, `src/eigenlab/thermal.py`.

### 32. Gibbs state

Add `gibbs(H, beta)`, which is `e^{−βH} / Tr e^{−βH}` from `eigh`.

For `H = (ω/2) Z`, `⟨Z⟩ = −tanh(βω/2)`. At `β = 0` the state is `I/d`. For a nondegenerate `H` and large `β` it approaches the ground projector, with trace distance below `1e-6` at `β = 50` for `transverse_ising(3, 1, 1)`.

### 33. The Ising partition function from its modes

Take `ε_k` as the `n` nonnegative eigenvalues of `transverse_ising_bdg_matrix`. Then

```text
Z = Σ_ν e^{−β Σ_k (ν_k − 1/2) ε_k} = Π_k 2 cosh(β ε_k / 2)
```

For `n = 2, …, 6` and `β` in `{0.1, 1, 3}`, it matches `Tr e^{−βH}` to `1e-8` relative. Do the same for the Kitaev chain with `kitaev_bdg_matrix`.

### 34. Thermodynamics, consistently

Add `von_neumann_entropy(rho)` in bits.

For `transverse_ising(4, 1, 0.7)` at `β = 1.3`:

- `U = −∂ ln Z / ∂β`, by central difference with step `1e-5`, equals `Tr ρH` to `1e-8`.
- With `F = −ln Z / β`, the entropy `β(U − F)` in nats equals `von_neumann_entropy(gibbs(H, β)) · ln 2` to `1e-8`.

## Free-fermion entanglement

### 35. Entropy at any cut

Add `bipartite_entropy(state, cut)` in bits, for the first `cut` qubits against the rest.

At the middle cut it equals `entanglement_entropy`. `ghz(n)` has 1 bit at every cut. A product state has 0.

### 36. Peschel's correlation matrix

Take the open tight-binding chain at `n = 6` and `n = 8`, half filled with `N = n/2`. The single-particle energies `−2t cos(πk/(n+1))` have no zero and no degeneracy for even `n`, so the ground state is unique. Take it from the `N = n/2` block and embed it back.

The test builds `C_ij = Σ_{k occupied} φ_k(i) φ_k(j)` from the single-particle matrix itself. For each cut `ℓ`, the eigenvalues `ν` of `C` restricted to sites `0 … ℓ−1` give

```text
S = −Σ [ν log₂ ν + (1−ν) log₂ (1−ν)]
```

which equals `bipartite_entropy` of the many-body ground state at `ℓ` to `1e-8`.

This works because a contiguous left block of Jordan–Wigner qubits carries no string from outside the block.

`one_particle_rdm` of that ground state, restricted to the block, equals `C_A` to `1e-10`.

### 37. SSH dimers

At `v = 1`, `w = 0`, `n = 4`, `N = 2`, each strong bond holds one fermion in its bonding orbital, `(|10⟩ + |01⟩)/√2`.

Cutting through a strong bond (`ℓ = 1`) gives 1 bit. Cutting between dimers (`ℓ = 2`) gives 0. The Peschel formula agrees at both cuts.

Do not use `v = 0`, `w = 1` at `N = 2`. The end sites decouple, and that ground space is degenerate.

## The scope line

### 38. What this haul added

When phases 1–37 are green, update `docs/scope.md` and the README so these are described as present:

- STO-3G integrals over s Gaussians
- restricted Hartree–Fock
- the H₂ curve at any bond length, with its gradient, the Pulay term, the equilibrium bond length, and the harmonic frequency
- HeH⁺ at any bond length
- H₄ in eight qubits
- the bit-flip, phase-flip, five-qubit, and Steane codes, with the Knill–Laflamme check
- Gibbs states and the free-fermion partition function
- entropy at any cut, and Peschel's correlation-matrix entanglement

Move "A derivative of the H₂ curve" and "Any molecule beyond H₂ and HeH⁺" off the not-done list. Leave these named as not done:

- Any molecule beyond H₂, HeH⁺, and H₄.
- Any basis beyond STO-3G s functions.
- Exact diagonalization above eight qubits.

## Done

After phase 38 the molecules are functions, not tables, and every number in them comes from an integral this repo computes. No new dependency. The eight-qubit diagonalization limit is where it was.
