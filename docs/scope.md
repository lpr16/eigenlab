# Scope

Eigenlab is exact diagonalization and state arithmetic for small quantum systems.

In scope:

- Kets, density matrices, purity, partial trace, purification, Bloch vectors, measurement probabilities.
- GHZ and W states.
- Pauli strings, expectation values, algebraic Pauli multiplication without matrices, commutators and anticommutators.
- A Pauli string applied to a ket as a signed permutation, Pauli conjugation, and Clifford conjugation by `H` and `S`.
- Two-qubit entanglement: Schmidt spectrum, entanglement entropy, concurrence, negativity, entanglement of formation.
- Pauli measurement: outcome probabilities, post-measurement states, Shannon entropy, the Helstrom bound, the swap test.
- Circuits with an exact output: one Grover iteration, the quantum Fourier transform, the clock shift, the CHSH operator, quantum relative entropy.
- Spin Hamiltonians with closed-form spectra: Heisenberg dimer, XXZ dimer, transverse-field Ising.
- Symmetry blocks: the parity of the transverse-field Ising chain and its two blocks, the occupation block of the Hubbard dimer, the spin blocks of the Heisenberg dimer.
- Exact unitary evolution, Trotter product formula, Strang splitting, the fourth-order Yoshida product, and their error scaling.
- The Loschmidt echo, the spin echo, and the Ehrenfest derivative of a Pauli expectation.
- One-qubit channels: amplitude damping and dephasing as Kraus maps and closed-form master-equation solutions, the depolarizing channel, and the Choi matrix.
- Fermionic operators, Jordan–Wigner and Bravyi–Kitaev transformations, open tight-binding chains, Hubbard dimer.
- Quadratic fermions: the Su–Schrieffer–Heeger chain, the Kitaev chain and its Bogoliubov spectrum, and the periodic tight-binding ring with its Jordan–Wigner closing string, in either encoding.
- The open transverse-field Ising chain spectrum from its free-fermion Bogoliubov matrix.
- Particle and spin sector tags, total spin `S²`, one-particle density matrix, and natural occupations, with sector tags and one-particle RDMs in either encoding.
- Hellmann–Feynman derivatives for the XXZ dimer, transverse-field Ising, the Hubbard dimer, and the Kitaev chemical potential.
- Nondegenerate and degenerate perturbation theory, an avoided crossing, a discrete Berry phase, and Lanczos with full reorthogonalization.
- Second-quantized molecular Hamiltonians from 1- and 2-electron integrals in chemist notation: STO-3G H₂ potential energy curve and STO-3G HeH⁺, checked against published full-CI literature.

Not done:

- A derivative of the H₂ curve.
- Any molecule beyond H₂ and HeH⁺.
- Exact diagonalization above eight qubits.

A Pauli expectation on a ket above eight qubits permutes amplitudes. That is not exact diagonalization, and it does not close the third item.

Out of scope: finance, variational-algorithm leaderboards, claims about hardware advantage.
