# eigenlab

A small lab for quantum science. The object of study is a state and the spectrum of a Hamiltonian, not a market.

Current hardware is irrelevant here. Two to eight qubits, exact linear algebra, and a number you can differentiate by hand.

## What lives here

| Track | Question | Objects |
| --- | --- | --- |
| States | What does a ket predict? | Bloch vectors, Bell pairs, purity, partial trace, concurrence |
| Operators | How do Paulis compose? | `X`, `Y`, `Z`, Kronecker products, expectations, algebra without matrices |
| Spectra | What are the energies? | Heisenberg dimer, XXZ dimer, transverse-field Ising, free-fermion Bogoliubov spectrum, eigensystem |
| Dynamics | How does a state evolve? | Exact spectral unitary, first-order Trotter, Strang splitting, error scaling, Hellmann–Feynman derivatives |
| Fermions | How do electrons hop? | Jordan–Wigner and Bravyi–Kitaev transformations, tight-binding chain, Hubbard dimer, sector tagging, total spin `S²`, natural occupations |
| Molecules | What is the chemical ground state? | Integral Hamiltonian in chemist notation, STO-3G H₂ potential energy curve, STO-3G HeH⁺ |

Bravyi–Kitaev, the free-fermion transverse Ising Bogoliubov spectrum, total spin `S²`, natural occupations, the three Hellmann–Feynman checks, and STO-3G HeH⁺ are implemented and tested. Left as not done: a derivative of the H₂ curve, any molecule beyond H₂ and HeH⁺, and exact diagonalization above eight qubits. No heavy SDKs; NumPy and standard linear algebra are enough to be right in public.

## Run

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT.
