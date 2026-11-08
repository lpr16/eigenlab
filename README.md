# eigenlab

A small lab for quantum science. The object of study is a state and the spectrum of a Hamiltonian, not a market.

Current hardware is irrelevant here. Two to eight qubits, exact linear algebra, and a number you can differentiate by hand.

## What lives here

| Track | Question | Objects |
| --- | --- | --- |
| States | What does a ket predict? | Bloch vectors, Bell pairs, GHZ and W states, purity, partial trace, purification, entropy at any cut, concurrence, negativity, entanglement of formation, relative entropy, mutual information, conditional entropy, the Holevo quantity, quantum Fisher information |
| Measurement | What does a reading return? | Pauli outcome probabilities, post-measurement states, Shannon entropy, Helstrom bound, swap test, CHSH |
| Circuits | What does a short circuit output? | One Grover iteration, the quantum Fourier transform, the clock shift, teleportation, superdense coding, phase estimation |
| Operators | How do Paulis compose? | `X`, `Y`, `Z`, Kronecker products, expectations, algebra without matrices, commutators, a Pauli as a signed permutation, Pauli and Clifford conjugation |
| Spectra | What are the energies? | Heisenberg dimer and ring, translation symmetry, momentum blocks, one-magnon band, XXZ dimer, transverse-field Ising, exact modes of the critical Ising chain, free-fermion Bogoliubov spectrum, parity, occupation and spin blocks, eigensystem, Lanczos |
| Perturbation | How do the energies move? | Nondegenerate and degenerate perturbation theory, avoided crossing, discrete Berry phase, Hellmann–Feynman derivatives |
| Dynamics | How does a state evolve? | Exact spectral unitary, first-order Trotter, Strang splitting, fourth-order Yoshida product, time-dependent midpoint evolution with the exact rotating-drive solution, error scaling, Loschmidt echo, spin echo, Ehrenfest |
| Channels | What does noise do to a qubit? | Amplitude damping and dephasing as Kraus maps and in closed form, depolarizing channel, Choi matrix, the Lindblad equation, its steady states, detailed balance, and entanglement sudden death |
| Codes | How is a logical qubit protected? | The bit-flip, phase-flip, five-qubit, and Steane codes, the toric code on a 2 × 2 torus, with code distance and the Knill–Laflamme check |
| Thermal | What is the equilibrium state? | Gibbs states, von Neumann entropy, trace distance, and the free-fermion partition function |
| Fermions | How do electrons hop? | Jordan–Wigner, Bravyi–Kitaev, and parity encodings, single-qubit `Z` tapering, tight-binding chain and ring with its Jordan–Wigner closing string, SSH chain, Kitaev chain and its Bogoliubov spectrum, Hubbard dimer, sector tagging and one-particle RDMs, total spin `S²`, natural occupations, Peschel's correlation-matrix entanglement |
| Molecules | What is the chemical ground state? | STO-3G integrals over s Gaussians, restricted Hartree–Fock, the H₂ curve at any bond length (with its gradient, the Pulay term, the equilibrium bond length, the harmonic frequency, and tapering to two qubits), HeH⁺ at any bond length (and tapered to three qubits), and H₄ in eight qubits |
| Oscillator | How does a qubit couple to a mode? | A truncated oscillator, coherent states, and the Jaynes–Cummings model with vacuum Rabi oscillations |

Everything in the table is implemented and tested, including the parity encoding and single-qubit `Z` tapering (with H₂ on two qubits and HeH⁺ on three), the exact modes of the critical Ising chain, translation symmetry, momentum blocks, the Heisenberg ring and its one-magnon band, the toric code on a 2 × 2 torus, teleportation, superdense coding, phase estimation, mutual information, conditional entropy, the Holevo quantity, quantum Fisher information, time-dependent evolution with the exact rotating-drive solution, the Lindblad equation, its steady states, detailed balance, entanglement sudden death, and a truncated oscillator, coherent states, and the Jaynes–Cummings model.

Not done:

- Any molecule beyond H₂, HeH⁺, and H₄.
- Any basis beyond STO-3G s functions.
- Exact diagonalization above eight qubits.

A Pauli expectation on a larger ket permutes amplitudes; that is not diagonalization. No heavy SDKs; NumPy and standard linear algebra are enough to be right in public.

## Run

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT.
