# Scope

Eigenlab is exact diagonalization and state arithmetic for small quantum systems.

In scope:

- Kets, density matrices, purity, partial trace, purification, Bloch vectors, measurement probabilities.
- GHZ and W states.
- Pauli strings as bit vectors, GF(2) rank and nullspace, expectation values, algebraic Pauli multiplication without matrices, commutators and anticommutators.
- A Pauli string applied to a ket as a signed permutation, Pauli conjugation, and Clifford conjugation by `H` and `S`.
- Two-qubit entanglement: Schmidt spectrum, entanglement entropy, concurrence, negativity, entanglement of formation, and bipartite entropy at any cut.
- Mutual information, conditional entropy, the Holevo quantity, and quantum Fisher information of pure and mixed states.
- Pauli measurement: outcome probabilities, post-measurement states, Shannon entropy, the Helstrom bound, the swap test.
- Circuits with an exact output: one Grover iteration, the quantum Fourier transform, the clock shift, the CHSH operator, quantum relative entropy, teleportation (including through a Werner pair), superdense coding, and phase estimation.
- Bell inequalities by enumeration: the Mermin operator, the Peres–Mermin square, and the Horodecki CHSH criterion.
- Spin Hamiltonians with closed-form spectra: Heisenberg dimer, XXZ dimer, transverse-field Ising, and the Heisenberg ring with translation symmetry, momentum blocks, and its one-magnon band.
- Symmetry blocks: the parity of the transverse-field Ising chain and its two blocks, the occupation block of the Hubbard dimer, the spin blocks of the Heisenberg dimer.
- Exact unitary evolution, Trotter product formula, Strang splitting, the fourth-order Yoshida product, and their error scaling.
- Time-dependent midpoint evolution with the exact rotating-drive solution, its Floquet quasienergies, and second-order scaling.
- The Loschmidt echo, the spin echo, and the Ehrenfest derivative of a Pauli expectation.
- One-qubit channels: amplitude damping and dephasing as Kraus maps and closed-form master-equation solutions, the depolarizing channel, and the Choi matrix.
- Open quantum systems: scaling-and-squaring matrix exponential, the Lindblad equation, its steady states, detailed balance giving the Gibbs state, and entanglement sudden death.
- Stabilizer error-correcting codes: the bit-flip, phase-flip, five-qubit, and Steane codes, the toric code on tori up to 5 × 5, with code distance and the Knill–Laflamme check.
- Stabilizer tableaux with exact measurement and expectations, including a 64-qubit GHZ state.
- The one-qubit Clifford group and its 2-design checks, twirling, exactly averaged randomized benchmarking, the T gate in the Clifford hierarchy, and the stabilizer Rényi entropy.
- Thermal states and statistical mechanics: Gibbs states, von Neumann entropy, trace distance, and the free-fermion partition function.
- Fermionic operators, Jordan–Wigner, Bravyi–Kitaev, and parity transformations, open tight-binding chains, the Hubbard dimer, and the open Hubbard chain through the Heisenberg limit at large U.
- Z₂ symmetry tapering, with H₂ on one qubit and HeH⁺ on three.
- Collective spin, Dicke states, one-axis twisting, and the squeezing bound on the Fisher information, including the symmetric subspace at N = 64.
- Quadratic fermions: the Su–Schrieffer–Heeger chain, the Kitaev chain and its Bogoliubov spectrum, and the periodic tight-binding ring with its Jordan–Wigner closing string, in any of the three encodings.
- The open transverse-field Ising chain spectrum from its free-fermion Bogoliubov matrix, and the exact modes of the critical Ising chain.
- Particle and spin sector tags, total spin `S²`, one-particle density matrix, natural occupations, and Peschel's correlation-matrix entanglement, with sector tags and one-particle RDMs across encodings.
- Hellmann–Feynman derivatives for the XXZ dimer, transverse-field Ising, the Hubbard dimer, and the Kitaev chemical potential.
- Nondegenerate and degenerate perturbation theory, an avoided crossing, a discrete Berry phase, and Lanczos with full reorthogonalization.
- STO-3G integrals over s Gaussians (overlap, kinetic, Boys nuclear attraction, two-electron repulsion), the 6-31G basis for hydrogen, Löwdin orthogonalization, and restricted Hartree–Fock.
- Second-quantized molecular Hamiltonians from 1- and 2-electron integrals in chemist notation: the H₂ curve at any bond length, with its gradient, the Pulay term, the equilibrium bond length, and the harmonic frequency; HeH⁺ at any bond length; H₄ in eight qubits; and H₂ in 6-31G, tapered by its Z₂ symmetries.
- A truncated harmonic oscillator, coherent states, and the Jaynes–Cummings model with vacuum Rabi oscillations.

Not done:

- Any molecule beyond H₂, HeH⁺, and H₄.
- Any basis with p or higher functions, and 6-31G for any atom but hydrogen.
- Exact diagonalization above eight qubits.

A Pauli expectation on a ket above eight qubits permutes amplitudes. A tableau on 64 qubits and a symmetric-subspace block at N = 64 are not exact diagonalization above eight qubits, and they do not close the third item.

Out of scope: finance, variational-algorithm leaderboards, claims about hardware advantage.
