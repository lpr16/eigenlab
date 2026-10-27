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


def test_noncommuting_trotter_and_strang_error():
    # Phase 13:
    # Two-site transverse Ising: H = -J ZZ - h (XI + IX), where [ZZ, XI] != 0
    from eigenlab.dynamics import strang_unitary, trotter_unitary, unitary
    from eigenlab.hamiltonian import transverse_ising_terms

    h = transverse_ising_terms(2, coupling=1.0, field=1.0)

    # Step sizes: several steps with successive halving
    dt_list = [0.1, 0.05, 0.025, 0.0125]
    err_first_order: list[float] = []
    err_strang: list[float] = []

    for dt in dt_list:
        u_exact = unitary(h, dt)
        u_trot = trotter_unitary(h, dt)
        u_strang = strang_unitary(h, dt)

        # Frobenius distance: ||U_approx - U_exact||_F
        dist_trot = float(np.linalg.norm(u_trot - u_exact, ord="fro"))
        dist_strang = float(np.linalg.norm(u_strang - u_exact, ord="fro"))

        err_first_order.append(dist_trot)
        err_strang.append(dist_strang)

    # 1. Halving Δt cuts first-order distance by about 4: (Δt/2)² / Δt² = 1/4 => ratio ≈ 4
    for i in range(len(dt_list) - 1):
        ratio_first = err_first_order[i] / err_first_order[i + 1]
        assert np.isclose(ratio_first, 4.0, atol=0.05)

    # 2. Halving Δt cuts Strang distance by about 8: (Δt/2)³ / Δt³ = 1/8 => ratio ≈ 8
    for i in range(len(dt_list) - 1):
        ratio_strang = err_strang[i] / err_strang[i + 1]
        assert np.isclose(ratio_strang, 8.0, atol=0.05)

    # 3. Log-log slope check with tight fixed tolerance around 2 and 3
    # log(error) ≈ p * log(dt) + c
    log_dts = np.log(dt_list)
    slope_first = float(np.polyfit(log_dts, np.log(err_first_order), 1)[0])
    slope_strang = float(np.polyfit(log_dts, np.log(err_strang), 1)[0])

    # Tight tolerance around 2 and 3
    assert np.isclose(slope_first, 2.0, atol=0.01)
    assert np.isclose(slope_strang, 3.0, atol=0.01)


def test_loschmidt_echo():
    # Phase 34: Loschmidt echo
    # H = (ω/2) Z on |+⟩: |ψ(t)⟩ = (e^{−iωt/2}|0⟩ + e^{+iωt/2}|1⟩)/√2,
    # ⟨+|ψ(t)⟩ = (e^{−iωt/2} + e^{+iωt/2}) / 2 = cos(ωt/2).
    from eigenlab.dynamics import loschmidt
    from eigenlab.pauli import Z

    omega = 1.0
    h = 0.5 * omega * Z
    assert np.isclose(loschmidt(h, PLUS, 0.0), 1.0, atol=1e-12)
    assert np.isclose(loschmidt(h, PLUS, np.pi / omega), 0.0, atol=1e-12)
    assert np.isclose(loschmidt(h, PLUS, 1.0), np.cos(omega * 1.0 / 2.0) ** 2, atol=1e-12)


def test_spin_echo():
    # Phase 35: Spin echo
    # X Z X = −Z, so X exp(−iHτ) X = exp(+iHτ) for H = (ω/2) Z, and
    #   exp(−iHτ) X exp(−iHτ) = X (X exp(−iHτ) X) exp(−iHτ) = X exp(+iHτ) exp(−iHτ) = X.
    # X|+⟩ = |+⟩, so the echo returns |+⟩.
    from eigenlab.dynamics import spin_echo, unitary
    from eigenlab.pauli import X, Z

    omega = 1.3
    tau = 0.7
    h = 0.5 * omega * Z

    u = unitary(h, tau)
    assert np.allclose(X @ u @ X, u.conj().T, atol=1e-12)
    assert np.allclose(u @ X @ u, X, atol=1e-12)

    final = spin_echo(h, PLUS, tau)
    assert np.allclose(final, evolve(h, X @ evolve(h, PLUS, tau), tau), atol=1e-12)
    assert np.isclose(abs(np.vdot(PLUS, final)), 1.0, atol=1e-12)
    assert np.isclose(expectation(X, final), 1.0, atol=1e-12)

    # Without the pulse, ⟨X(2τ)⟩ = cos(ω · 2τ), which is not 1.
    free = evolve(h, PLUS, 2.0 * tau)
    assert np.isclose(expectation(X, free), np.cos(omega * 2.0 * tau), atol=1e-12)
    assert not np.isclose(expectation(X, free), 1.0, atol=1e-3)


def test_ehrenfest():
    # Phase 36: Ehrenfest
    # H = (ω/2) Z on |+⟩: ⟨X(t)⟩ = cos(ωt), so d⟨X⟩/dt = −ω sin(ωt).
    # Heisenberg: d⟨X⟩/dt = i ⟨[H, X]⟩ = i (ω/2) ⟨[Z, X]⟩ = i (ω/2) (2i) ⟨Y⟩ = −ω ⟨Y⟩.
    from eigenlab.pauli import Z, expectation_label, pauli_comm

    phase, label = pauli_comm("Z", "X")
    assert label == "Y" and np.isclose(phase, 2j)

    omega = 1.7
    h = 0.5 * omega * Z
    step = 1e-6
    for t in [0.0, 0.3, 1.0]:
        x_plus = expectation_label(evolve(h, PLUS, t + step), "X")
        x_minus = expectation_label(evolve(h, PLUS, t - step), "X")
        central = (x_plus - x_minus) / (2.0 * step)

        closed_form = -omega * np.sin(omega * t)
        commutator = complex(1j * (omega / 2.0) * phase * expectation_label(evolve(h, PLUS, t), label))
        assert abs(commutator.imag) < 1e-12
        assert abs(central - closed_form) < 1e-8
        assert abs(central - commutator.real) < 1e-8


def test_two_constant_pieces():
    # Phase 37: Two constant pieces
    # [ZI, IZ] = 0, so exp(−i(ZI + IZ)t) = exp(−i IZ t) exp(−i ZI t): composing evolve is exact.
    # [ZI, XI] = 2i YI ≠ 0, so the composition misses the commutator at order t².
    from eigenlab.hamiltonian import Hamiltonian

    t = 0.4
    psi = np.kron(PLUS, PLUS)

    def ray_distance(a: np.ndarray, b: np.ndarray) -> float:
        # min over φ of ‖a − e^{iφ} b‖ for unit kets
        return float(np.sqrt(max(0.0, 2.0 - 2.0 * abs(np.vdot(a, b)))))

    zi = Hamiltonian([(1.0, "ZI")])
    iz = Hamiltonian([(1.0, "IZ")])
    xi = Hamiltonian([(1.0, "XI")])

    composed = evolve(iz, evolve(zi, psi, t), t)
    assert np.allclose(composed, evolve(zi + iz, psi, t), atol=1e-12)

    composed = evolve(xi, evolve(zi, psi, t), t)
    assert ray_distance(composed, evolve(zi + xi, psi, t)) > 1e-3


def test_yoshida_fourth_order():
    # Phase 38: Fourth-order product formula
    # S2 is symmetric, so its error is odd in Δt: S2(Δt) = exp(−iHΔt + Δt³ E3 + O(Δt⁵)).
    # With 2θ³ + (1 − 2θ)³ = 0, i.e. θ = 1 / (2 − 2^{1/3}), the Δt³ terms cancel in
    # S2(θΔt) S2((1 − 2θ)Δt) S2(θΔt), leaving a one-step error O(Δt⁵): halving Δt cuts it by 32.
    from eigenlab.dynamics import strang_unitary, unitary, yoshida_unitary
    from eigenlab.hamiltonian import transverse_ising_terms

    theta = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
    assert np.isclose(2.0 * theta**3 + (1.0 - 2.0 * theta) ** 3, 0.0, atol=1e-12)
    assert 1.0 - 2.0 * theta < 0.0

    # Two-site transverse Ising: [ZZ, XI] ≠ 0.
    h = transverse_ising_terms(2, coupling=1.0, field=1.0)

    # strang_unitary accepts a negative step: S2(−Δt) S2(Δt) = I.
    assert np.allclose(strang_unitary(h, -0.3) @ strang_unitary(h, 0.3), np.eye(4), atol=1e-12)

    dt_list = [0.1, 0.05, 0.025, 0.0125]
    errors = [
        float(np.linalg.norm(yoshida_unitary(h, dt) - unitary(h, dt), ord="fro"))
        for dt in dt_list
    ]

    for i in range(len(dt_list) - 1):
        assert np.isclose(errors[i] / errors[i + 1], 32.0, atol=1.5)

    slope = float(np.polyfit(np.log(dt_list), np.log(errors), 1)[0])
    assert np.isclose(slope, 5.0, atol=0.05)
