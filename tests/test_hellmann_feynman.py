import numpy as np

from eigenlab.fermions import (
    hubbard_dimer,
    number_operator,
    total_number_operator,
    total_spin_z,
)
from eigenlab.hamiltonian import (
    eigensystem,
    spectrum,
    transverse_ising,
    xxz_dimer_terms,
)
from eigenlab.pauli import expectation, pauli
from eigenlab.states import basis, bell


def test_hellmann_feynman_xxz_dimer():
    # Phase 6: XXZ dimer, H = XX + YY + Δ ZZ, so ∂H/∂Δ = ZZ.
    # On the singlet (|01⟩ − |10⟩)/√2, ⟨ZZ⟩ = −1, and the energy −2 − Δ has the same slope (−1).
    singlet = bell("psi-")
    zz = pauli("ZZ")
    exp_zz_singlet = expectation(zz, singlet)
    assert np.isclose(exp_zz_singlet, -1.0, atol=1e-12)

    # For Δ = 1.0 (Heisenberg point), energy is -3. Slope dE/dΔ = -1.
    delta = 1.5
    h_xxz = xxz_dimer_terms(delta=delta)
    assert np.isclose(h_xxz.expectation(singlet), -2.0 - delta, atol=1e-12)

    # On |00⟩, ⟨ZZ⟩ = 1 and the energy is Δ (slope +1).
    state_00 = basis(0, 2)
    exp_zz_00 = expectation(zz, state_00)
    assert np.isclose(exp_zz_00, 1.0, atol=1e-12)
    assert np.isclose(h_xxz.expectation(state_00), delta, atol=1e-12)


def test_hellmann_feynman_transverse_ising():
    # Phase 6: Open transverse Ising, n = 3, J = 1, h = 1.
    # ∂H/∂h = −∑_i X_i.
    # A central difference of the ground energy with step 10^{-4} matches that expectation to 10^{-6}.
    n = 3
    j_val = 1.0
    h_val = 1.0

    _, vectors = eigensystem(transverse_ising(n, coupling=j_val, field=h_val))
    ground_state = vectors[:, 0]

    sum_x = pauli("XII") + pauli("IXI") + pauli("IIX")
    dh_dh = -sum_x
    exp_dh_dh = expectation(dh_dh, ground_state)

    # Central difference with step 1e-4
    eps = 1e-4
    e_plus = spectrum(transverse_ising(n, coupling=j_val, field=h_val + eps))[0]
    e_minus = spectrum(transverse_ising(n, coupling=j_val, field=h_val - eps))[0]
    central_diff = (e_plus - e_minus) / (2.0 * eps)

    assert abs(exp_dh_dh - central_diff) < 1e-6


def test_hellmann_feynman_hubbard_dimer():
    # Phase 6: Hubbard dimer, t = 1, on the half-filled singlet rather than the full-space ground state.
    # E(U) = (U − √(U² + 16 t²)) / 2
    # dE/dU = (1 − U / √(U² + 16 t²)) / 2
    # and this equals ⟨n_{0↑} n_{0↓} + n_{1↑} n_{1↓}⟩.
    # Check U = 0, where the derivative is 1/2, and U = 4, from the algebraic expression.
    t = 1.0
    n_tot = total_number_operator(4)
    sz_tot = total_spin_z(2)
    double_occ_op = (number_operator(0, 4) @ number_operator(1, 4)) + (
        number_operator(2, 4) @ number_operator(3, 4)
    )

    for u in [0.0, 4.0]:
        h_hub = hubbard_dimer(t=t, U=u)
        evals, evecs = eigensystem(h_hub, commuting=[n_tot, sz_tot])

        expected_singlet_energy = (u - np.sqrt(u**2 + 16.0 * t**2)) / 2.0

        # Filter half-filled singlet (lowest state in N = 2, S_z = 0 sector)
        n2_sz0 = []
        for col in range(16):
            v = evecs[:, col]
            exp_n = n_tot.expectation(v)
            exp_sz = sz_tot.expectation(v)
            if np.isclose(round(exp_n), 2) and np.isclose(exp_sz, 0.0, atol=1e-8):
                n2_sz0.append((evals[col], v))

        n2_sz0.sort(key=lambda item: item[0])
        lowest_energy, singlet_vec = n2_sz0[0]
        assert np.isclose(lowest_energy, expected_singlet_energy, atol=1e-8)

        # Theoretical derivative: dE/dU = (1 − U / √(U² + 16 t²)) / 2
        dE_dU_exact = (1.0 - u / np.sqrt(u**2 + 16.0 * t**2)) / 2.0

        if u == 0.0:
            assert np.isclose(dE_dU_exact, 0.5, atol=1e-12)

        exp_double_occ = double_occ_op.expectation(singlet_vec)
        assert np.isclose(exp_double_occ, dE_dU_exact, atol=1e-8)
