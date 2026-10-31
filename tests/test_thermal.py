import numpy as np

from eigenlab.fermions import kitaev_bdg_matrix, kitaev_chain
from eigenlab.hamiltonian import eigensystem, transverse_ising, transverse_ising_bdg_matrix
from eigenlab.pauli import Z
from eigenlab.states import density
from eigenlab.thermal import (
    bdg_partition_function,
    gibbs,
    ising_partition_function,
    kitaev_partition_function,
    partition_function,
    trace_distance,
    von_neumann_entropy,
)


def test_gibbs_state():
    # Phase 32: Gibbs state
    # For H = (ω/2) Z, eigenvalues are +ω/2 on |0⟩ and −ω/2 on |1⟩, so
    #   ⟨Z⟩ = (e^{−βω/2} − e^{+βω/2}) / (e^{−βω/2} + e^{+βω/2}) = −tanh(βω / 2).
    for omega in [0.7, 1.0, 2.3]:
        h_1q = 0.5 * omega * Z
        for beta in [0.0, 0.25, 1.0, 3.5]:
            rho = gibbs(h_1q, beta)
            assert np.isclose(np.trace(rho), 1.0, atol=1e-12)
            assert np.allclose(rho, rho.conj().T, atol=1e-12)
            exp_z = float(np.real(np.trace(rho @ Z)))
            assert np.isclose(exp_z, -np.tanh(0.5 * beta * omega), atol=1e-12)

    # At β = 0 the state is I / d
    h_ising = transverse_ising(3, 1.0, 1.0)
    rho_0 = gibbs(h_ising, 0.0)
    assert np.allclose(rho_0, np.eye(8) / 8.0, atol=1e-12)

    # For nondegenerate H = transverse_ising(3, 1, 1) and β = 50,
    # trace distance to the ground projector is below 1e-6.
    evals, evecs = eigensystem(h_ising)
    assert evals[1] - evals[0] > 0.1
    ground_proj = density(evecs[:, 0])
    rho_50 = gibbs(h_ising, 50.0)
    assert trace_distance(rho_50, ground_proj) < 1e-6


def test_ising_and_kitaev_partition_function_from_modes():
    # Phase 33: The Ising partition function from its modes
    # Z = Σ_ν e^{−β Σ_k (ν_k − 1/2) ε_k} = Π_k 2 cosh(β ε_k / 2).
    # For n = 2, …, 6 and β in {0.1, 1, 3}, matches Tr e^{−βH} to 1e-8 relative
    # for both transverse_ising and kitaev_chain.
    for n in range(2, 7):
        h_ising = transverse_ising(n, 1.0, 0.8)
        bdg_ising = transverse_ising_bdg_matrix(n, coupling=1.0, field=0.8)
        eps_ising = np.linalg.eigvalsh(bdg_ising)[n:]
        assert np.all(eps_ising >= -1e-12)

        h_kit = kitaev_chain(n, t=0.9, delta=1.2, mu=0.35)
        bdg_kit = kitaev_bdg_matrix(n, t=0.9, delta=1.2, mu=0.35)
        eps_kit = np.linalg.eigvalsh(bdg_kit)[n:]
        assert np.all(eps_kit >= -1e-12)

        for beta in [0.1, 1.0, 3.0]:
            z_ising_exact = partition_function(h_ising, beta)
            z_ising_modes = float(np.prod(2.0 * np.cosh(0.5 * beta * eps_ising)))
            assert np.isclose(z_ising_modes, z_ising_exact, rtol=1e-8, atol=0.0)
            assert np.isclose(
                ising_partition_function(n, beta, coupling=1.0, field=0.8),
                z_ising_exact,
                rtol=1e-8,
                atol=0.0,
            )

            z_kit_exact = partition_function(h_kit, beta)
            z_kit_modes = float(np.prod(2.0 * np.cosh(0.5 * beta * eps_kit)))
            assert np.isclose(z_kit_modes, z_kit_exact, rtol=1e-8, atol=0.0)
            assert np.isclose(
                kitaev_partition_function(n, beta, t=0.9, delta=1.2, mu=0.35),
                z_kit_exact,
                rtol=1e-8,
                atol=0.0,
            )
            assert np.isclose(bdg_partition_function(bdg_kit, beta), z_kit_exact, rtol=1e-8, atol=0.0)


def test_thermodynamics_consistently():
    # Phase 34: Thermodynamics, consistently
    # For transverse_ising(4, 1, 0.7) at β = 1.3:
    # - U = −∂ ln Z / ∂β (central difference step 1e-5) equals Tr(ρ H) to 1e-8.
    # - With F = −ln Z / β, β(U − F) in nats equals von_neumann_entropy(gibbs(H, β)) · ln 2 to 1e-8.
    H = transverse_ising(4, 1.0, 0.7)
    beta = 1.3
    step = 1e-5

    rho = gibbs(H, beta)
    u_trace = float(np.real(np.trace(rho @ H)))

    ln_z_plus = np.log(partition_function(H, beta + step))
    ln_z_minus = np.log(partition_function(H, beta - step))
    u_diff = -(ln_z_plus - ln_z_minus) / (2.0 * step)
    assert np.isclose(u_diff, u_trace, atol=1e-8)

    ln_z = np.log(partition_function(H, beta))
    f_free = -ln_z / beta
    s_nats = beta * (u_trace - f_free)
    s_vne_bits = von_neumann_entropy(rho)
    assert np.isclose(s_nats, s_vne_bits * np.log(2.0), atol=1e-8)

    # Pure state has 0 bits; maximally mixed 1-qubit state I/2 has 1 bit
    assert von_neumann_entropy(np.array([1.0, 0.0])) == 0.0
    assert np.isclose(von_neumann_entropy(np.eye(2) / 2.0), 1.0, atol=1e-12)


