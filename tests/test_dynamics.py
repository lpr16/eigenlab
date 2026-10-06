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


def test_one_spin_closed_form():
    # Phase 11:
    # Hamiltonian H = (ω/2) Z, starting in |+⟩ = (|0⟩ + |1⟩)/√2.
    # Schrödinger equation: i d|ψ⟩/dt = H|ψ⟩
    # |ψ(t)⟩ = exp(-i H t) |+⟩ = (exp(-i ω t / 2)|0⟩ + exp(+i ω t / 2)|1⟩)/√2.
    #
    # Derivation of ⟨X(t)⟩:
    # ⟨X(t)⟩ = ⟨ψ(t)|X|ψ(t)⟩ = (exp(i ω t) + exp(-i ω t)) / 2 = cos(ω t).
    #
    # Derivation of ⟨Y(t)⟩:
    # Ehrenfest / Heisenberg equation:
    # d⟨X⟩/dt = i ⟨[H, X]⟩ = i ⟨[(ω/2)Z, X]⟩ = i (ω/2) (2i ⟨Y⟩) = -ω ⟨Y⟩.
    # Since ⟨X(t)⟩ = cos(ω t), d⟨X⟩/dt = -ω sin(ω t).
    # Therefore -ω ⟨Y(t)⟩ = -ω sin(ω t)  ==>  ⟨Y(t)⟩ = +sin(ω t).
    #
    # Alternatively via wave function:
    # Y|ψ(t)⟩ = (-i exp(+i ω t / 2)|0⟩ + i exp(-i ω t / 2)|1⟩)/√2
    # ⟨ψ(t)|Y|ψ(t)⟩ = (-i exp(i ω t) + i exp(-i ω t)) / 2
    #              = -i (2i sin(ω t)) / 2 = +sin(ω t).
    from eigenlab.pauli import X, Y, Z

    for omega in [1.0, 2.5, 0.4]:
        h_spin = 0.5 * omega * Z
        psi0 = PLUS

        # Check at arbitrary t
        for t in [0.2, 0.8, 1.7]:
            psi_t = evolve(h_spin, psi0, t)
            assert np.isclose(expectation(X, psi_t), np.cos(omega * t))
            assert np.isclose(expectation(Y, psi_t), np.sin(omega * t))

        # Check specific required points: t = 0, π/(2ω), π/ω
        # 1. t = 0: ⟨X(0)⟩ = 1, ⟨Y(0)⟩ = 0
        psi_0 = evolve(h_spin, psi0, 0.0)
        assert np.isclose(expectation(X, psi_0), 1.0)
        assert np.isclose(expectation(Y, psi_0), 0.0)

        # 2. t = π/(2ω): cos(π/2) = 0, sin(π/2) = +1
        psi_half = evolve(h_spin, psi0, np.pi / (2.0 * omega))
        assert np.isclose(expectation(X, psi_half), 0.0, atol=1e-12)
        assert np.isclose(expectation(Y, psi_half), 1.0, atol=1e-12)

        # 3. t = π/ω: cos(π) = -1, sin(π) = 0
        psi_pi = evolve(h_spin, psi0, np.pi / omega)
        assert np.isclose(expectation(X, psi_pi), -1.0, atol=1e-12)
        assert np.isclose(expectation(Y, psi_pi), 0.0, atol=1e-12)


def test_first_order_trotter_step():
    # Phase 12:
    # For a term list, one step is ∏ exp(-i c_k P_k Δt) using P² = I => cos(θ) I - i sin(θ) P
    from eigenlab.dynamics import trotter_step, trotter_unitary
    from eigenlab.hamiltonian import Hamiltonian

    # 1. One term matches evolve
    h_single = Hamiltonian([(1.3, "X")])
    psi_1q = (basis(0, 1) + 2.0j * basis(1, 1)) / np.sqrt(5)
    for dt in [0.05, 0.2, 0.8, 1.5]:
        psi_trot = trotter_step(h_single, psi_1q, dt)
        psi_ev = evolve(h_single, psi_1q, dt)
        assert np.allclose(psi_trot, psi_ev)
        # Unitary matrix also matches
        u_trot = trotter_unitary(h_single, dt)
        assert np.allclose(u_trot @ psi_1q, psi_ev)

    # 2. Two commuting terms: ZI and IZ match evolve at any Δt
    # Proof: [ZI, IZ] = 0, so exp(-i(a ZI + b IZ)Δt) = exp(-i a ZI Δt) exp(-i b IZ Δt) exactly
    h_comm1 = Hamiltonian([(1.2, "ZI"), (-0.7, "IZ")])
    psi_2q = (basis(0, 2) + basis(1, 2) - 1j * basis(3, 2)) / np.sqrt(3)
    for dt in [0.01, 0.1, 0.5, 1.0, 3.14, 10.0]:
        psi_trot = trotter_step(h_comm1, psi_2q, dt)
        psi_ev = evolve(h_comm1, psi_2q, dt)
        assert np.allclose(psi_trot, psi_ev)
        assert np.allclose(trotter_unitary(h_comm1, dt) @ psi_2q, psi_ev)

    # 3. Two commuting terms: ZZ and II match evolve at any Δt
    # Proof: [ZZ, II] = 0
    h_comm2 = Hamiltonian([(0.9, "ZZ"), (1.5, "II")])
    for dt in [0.01, 0.2, 1.0, 4.0]:
        psi_trot = trotter_step(h_comm2, psi_2q, dt)
        psi_ev = evolve(h_comm2, psi_2q, dt)
        assert np.allclose(psi_trot, psi_ev)
        assert np.allclose(trotter_unitary(h_comm2, dt) @ psi_2q, psi_ev)
