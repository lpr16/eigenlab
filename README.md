# eigenlab

A small lab for quantum science. The object of study is a state and the spectrum of a Hamiltonian, not a market.

Current hardware is irrelevant here. Two to eight qubits, exact linear algebra, and a number you can differentiate by hand.

## What lives here

| Track | Question | Objects |
| --- | --- | --- |
| States | What does a ket predict? | Bloch vectors, Bell pairs, GHZ and W states, purity, partial trace, purification, concurrence, negativity, entanglement of formation, relative entropy |
| Measurement | What does a reading return? | Pauli outcome probabilities, post-measurement states, Shannon entropy, Helstrom bound, swap test, CHSH |
| Circuits | What does a short circuit output? | One Grover iteration, the quantum Fourier transform, the clock shift |
| Operators | How do Paulis compose? | `X`, `Y`, `Z`, Kronecker products, expectations, algebra without matrices, commutators, a Pauli as a signed permutation, Pauli and Clifford conjugation |
| Spectra | What are the energies? | Heisenberg dimer, XXZ dimer, transverse-field Ising, free-fermion Bogoliubov spectrum, parity, occupation and spin blocks, eigensystem, Lanczos |
| Perturbation | How do the energies move? | Nondegenerate and degenerate perturbation theory, avoided crossing, discrete Berry phase, Hellmann–Feynman derivatives |
| Dynamics | How does a state evolve? | Exact spectral unitary, first-order Trotter, Strang splitting, fourth-order Yoshida product, error scaling, Loschmidt echo, spin echo, Ehrenfest |
| Channels | What does noise do to a qubit? | Amplitude damping and dephasing as Kraus maps and in closed form, depolarizing channel, Choi matrix |
| Fermions | How do electrons hop? | Jordan–Wigner and Bravyi–Kitaev transformations, tight-binding chain and ring with its Jordan–Wigner closing string, SSH chain, Kitaev chain and its Bogoliubov spectrum, Hubbard dimer, sector tagging and one-particle RDMs in either encoding, total spin `S²`, natural occupations |
| Molecules | What is the chemical ground state? | Integral Hamiltonian in chemist notation, STO-3G H₂ potential energy curve, STO-3G HeH⁺ |

Everything in the table is implemented and tested, including the Hellmann–Feynman checks for the XXZ dimer, transverse-field Ising, the Hubbard dimer, and the Kitaev chemical potential. Left as not done: a derivative of the H₂ curve, any molecule beyond H₂ and HeH⁺, and exact diagonalization above eight qubits. A Pauli expectation on a larger ket permutes amplitudes; that is not diagonalization. No heavy SDKs; NumPy and standard linear algebra are enough to be right in public.

## Run

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT.
