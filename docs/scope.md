# Scope

Eigenlab is exact diagonalization and state arithmetic for small quantum systems.

In scope:

- Kets, density matrices, purity, partial trace, Bloch vectors, measurement probabilities.
- Pauli strings, expectation values, algebraic Pauli multiplication without matrices.
- Two-qubit entanglement: Schmidt spectrum, entanglement entropy, concurrence.
- Spin Hamiltonians with closed-form spectra: Heisenberg dimer, XXZ dimer, transverse-field Ising.
- Exact unitary evolution, Trotter product formula, Strang splitting, and symplectic Trotter error scaling.
- Fermionic operators and Jordan–Wigner transformation, open tight-binding chains, Hubbard dimer, particle and spin sector tagging.
- Second-quantized molecular Hamiltonians from 1- and 2-electron integrals in chemist notation, and the STO-3G H₂ potential energy curve checked against published full-CI literature.

Next, specified in `docs/next-haul.md`:

- Bravyi–Kitaev, checked against the Jordan–Wigner spectra already computed.
- The open transverse-field Ising spectrum from its free-fermion Bogoliubov matrix.
- Total spin `S²`, the one-particle density matrix, and natural occupations.
- Hellmann–Feynman derivatives for the XXZ dimer, the transverse-field Ising, and the Hubbard dimer.
- One second molecule, HeH⁺ in STO-3G, from a named reference, on the same eigensolver.

Not in that haul: a force along the tabulated H₂ curve, a third molecule, and any system above eight qubits.

Out of scope: finance, variational-algorithm leaderboards, claims about hardware advantage.
