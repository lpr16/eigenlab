# eigenlab

A small lab for quantum science. The object of study is a state and the spectrum of a Hamiltonian, not a market.

Current hardware is irrelevant here. Two to eight qubits, exact linear algebra, and a number you can differentiate by hand.

## What lives here

| Track | Question | First object |
| --- | --- | --- |
| States | What does a ket predict? | Bloch vectors, Bell pairs, measurement probabilities |
| Operators | How do Paulis compose? | `X`, `Y`, `Z`, Kronecker products, expectations |
| Spectra | What are the energies? | Heisenberg dimer, transverse-field Ising, exact diagonalization |

Molecular Hamiltonians come later, through a fermion-to-qubit mapping checked against the same diagonalization. No SDK in the first commit. NumPy is enough to be wrong in public.

## Run

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT.
