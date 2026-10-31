import numpy as np

from eigenlab.hamiltonian import eigensystem, transverse_ising
from eigenlab.pauli import Z
from eigenlab.states import density
from eigenlab.thermal import gibbs, trace_distance


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
