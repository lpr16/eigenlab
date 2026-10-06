import numpy as np

from eigenlab.dynamics import evolve
from eigenlab.hamiltonian import heisenberg_dimer, transverse_ising
from eigenlab.pauli import expectation
from eigenlab.states import PLUS, ZERO, basis


def test_exact_evolution():
    # Phase 10:
    # 1. Spectral decomposition U = V exp(-i E t) V†
    # Norm and ⟨H⟩ are constant.
    h = heisenberg_dimer()
    psi0 = (basis(0, 2) + 2.0j * basis(1, 2) - basis(3, 2))
    psi0 = psi0 / np.linalg.norm(psi0)

    for t in [0.1, 0.5, 1.0, 2.3, 10.0]:
        psi_t = evolve(h, psi0, t)
        # Norm is constant
        assert np.isclose(np.linalg.norm(psi_t), 1.0)
        # ⟨H⟩ is constant
        assert np.isclose(expectation(h, psi_t), expectation(h, psi0))

    # 2. t = 0 leaves the state fixed
    assert np.allclose(evolve(h, psi0, 0.0), psi0)

    # 3. H = 0 leaves the state fixed (both as scalar and as zero matrix)
    assert np.allclose(evolve(0, psi0, 5.0), psi0)
    zero_mat = np.zeros_like(h)
    assert np.allclose(evolve(zero_mat, psi0, 5.0), psi0)

    # 4. Evolution by 2π under a Hamiltonian with integer spectrum returns the same ray.
    # Heisenberg dimer has integer spectrum {-3, 1, 1, 1}.
    # U(2π) = exp(-i (-3) 2π)|v0⟩⟨v0| + exp(-i (1) 2π) ∑ |vk⟩⟨vk|
    #       = exp(6π i)|v0⟩⟨v0| + exp(-2π i) ∑ |vk⟩⟨vk|
    #       = I
    # Thus |ψ(2π)⟩ = |ψ(0)⟩ (exact same ray, in fact exact same vector).
    psi_2pi = evolve(h, psi0, 2.0 * np.pi)
    assert np.allclose(psi_2pi, psi0)

    # Also check transverse Ising with J=1, h=0 (energies -2, 0, 2, all integers)
    ti_int = transverse_ising(3, coupling=1.0, field=0.0)
    psi_ti = (basis(0, 3) + basis(2, 3) + basis(7, 3)) / np.sqrt(3)
    psi_ti_2pi = evolve(ti_int, psi_ti, 2.0 * np.pi)
    assert np.allclose(psi_ti_2pi, psi_ti)
