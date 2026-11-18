# eigenlab

A small lab for quantum science. The object of study is a state and the spectrum of a Hamiltonian, not a market.

Current hardware is irrelevant here. Two to eight qubits, exact linear algebra, and a number you can differentiate by hand.

## What lives here

| Track | Question | Objects |
| --- | --- | --- |
| States | What does a ket predict? | Bloch vectors, Bell pairs, GHZ and W states, purity, partial trace, purification, entropy at any cut, concurrence, negativity, entanglement of formation, relative entropy, mutual information, conditional entropy, the Holevo quantity, quantum Fisher information |
| Measurement | What does a reading return? | Pauli outcome probabilities, post-measurement states, Shannon entropy, Helstrom bound, swap test, CHSH, the Mermin and Peres–Mermin inequalities, the Horodecki criterion |
| Circuits | What does a short circuit output? | One Grover iteration, the quantum Fourier transform, the clock shift, teleportation, superdense coding, phase estimation |
| Operators | How do Paulis compose? | `X`, `Y`, `Z`, Kronecker products, expectations, algebra without matrices, Pauli strings as bit vectors, GF(2) rank and nullspace, commutators, a Pauli as a signed permutation, Pauli and Clifford conjugation |
| Spectra | What are the energies? | Heisenberg dimer and ring, translation symmetry, momentum blocks, one-magnon band, XXZ dimer, transverse-field Ising, exact modes of the critical Ising chain, free-fermion Bogoliubov spectrum, parity, occupation and spin blocks, collective spin, Dicke states, eigensystem, Lanczos |
| Perturbation | How do the energies move? | Nondegenerate and degenerate perturbation theory, avoided crossing, discrete Berry phase, Hellmann–Feynman derivatives |
| Dynamics | How does a state evolve? | Exact spectral unitary, first-order Trotter, Strang splitting, fourth-order Yoshida product, time-dependent midpoint evolution with the exact rotating-drive solution and its Floquet quasienergies, one-axis twisting and the squeezing bound, error scaling, Loschmidt echo, spin echo, Ehrenfest |
| Channels | What does noise do to a qubit? | Amplitude damping and dephasing as Kraus maps and in closed form, depolarizing channel, Choi matrix, the Lindblad equation, its steady states, detailed balance, and entanglement sudden death |
| Codes | How is a logical qubit protected? | The bit-flip, phase-flip, five-qubit, and Steane codes, the toric code on tori up to 5 × 5, stabilizer tableaux (including a 64-qubit GHZ state), the one-qubit Clifford group, twirling, exactly averaged randomized benchmarking, the T gate, and the stabilizer Rényi entropy |
| Thermal | What is the equilibrium state? | Gibbs states, von Neumann entropy, trace distance, and the free-fermion partition function |
| Fermions | How do electrons hop? | Jordan–Wigner, Bravyi–Kitaev, and parity encodings, Z₂ tapering (H₂ on one qubit), tight-binding chain and ring with its Jordan–Wigner closing string, SSH chain, Kitaev chain and its Bogoliubov spectrum, Hubbard dimer and the Hubbard chain through its Heisenberg limit, sector tagging and one-particle RDMs, total spin `S²`, natural occupations, Peschel's correlation-matrix entanglement |
| Molecules | What is the chemical ground state? | STO-3G integrals over s Gaussians, 6-31G for hydrogen, restricted Hartree–Fock, the H₂ curve at any bond length (with its gradient, the Pulay term, the equilibrium bond length, the harmonic frequency, and tapering to one qubit), HeH⁺ at any bond length (and tapered to three qubits), H₄ in eight qubits, and H₂ in 6-31G |
| Oscillator | How does a qubit couple to a mode? | A truncated oscillator, coherent states, and the Jaynes–Cummings model with vacuum Rabi oscillations |

Everything in the table is implemented and tested. That includes bit-vector Pauli arithmetic and Z₂ tapering, stabilizer tableaux and a 64-qubit GHZ state, the toric code up to 5 × 5, the Clifford group, twirling, randomized benchmarking, the T gate, the stabilizer Rényi entropy, collective spin, Dicke states, one-axis twisting, the squeezing bound, the Mermin and Peres–Mermin inequalities, the Horodecki CHSH criterion, Floquet quasienergies, the Heisenberg limit of the Hubbard chain, and 6-31G hydrogen and H₂.

Not done:

- Any molecule beyond H₂, HeH⁺, and H₄.
- Any basis with p or higher functions, and 6-31G for any atom but hydrogen.
- Exact diagonalization above eight qubits.

A tableau on 64 qubits and a symmetric-subspace block at N = 64 are not exact diagonalization above eight qubits. No heavy SDKs; NumPy and standard linear algebra are enough to be right in public.

## Run

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT.
