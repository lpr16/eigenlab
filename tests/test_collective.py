import math

import numpy as np
import pytest

from eigenlab.collective import (
    collective_spin,
    dicke,
    one_axis_squeezing,
    one_axis_state,
    spin_matrices,
)
from eigenlab.dynamics import evolve
from eigenlab.states import PLUS, bipartite_entropy, ghz, qfi, w_state


def test_symmetric_subspace():
    # Phase 20: The symmetric subspace
    # Order the Dicke states as m = n/2 - k for k = 0 ... n.
    # Let V have columns dicke(n, k).
    # Then V† J_a V = spin_matrices(n/2) for every a in {x, y, z} and every n <= 8, to 1e-12.
    for n in range(1, 9):
        # Dicke states as columns of V
        cols = [dicke(n, k) for k in range(n + 1)]
        V = np.column_stack(cols)

        # Norm and orthogonality of Dicke states
        assert np.allclose(V.conj().T @ V, np.eye(n + 1), atol=1e-12)

        jx_h, jy_h, jz_h = collective_spin(n)
        jx_sp, jy_sp, jz_sp = spin_matrices(n / 2.0)

        # Dimension is n + 1
        assert jx_sp.shape == (n + 1, n + 1)
        assert jy_sp.shape == (n + 1, n + 1)
        assert jz_sp.shape == (n + 1, n + 1)

        # V† J_a V = spin_matrices(n/2) for every a and every n <= 8 to 1e-12
        assert np.allclose(V.conj().T @ jx_h @ V, jx_sp, atol=1e-12)
        assert np.allclose(V.conj().T @ jy_h @ V, jy_sp, atol=1e-12)
        assert np.allclose(V.conj().T @ jz_h @ V, jz_sp, atol=1e-12)

        # Also check with .to_matrix()
        assert np.allclose(V.conj().T @ jx_h.to_matrix() @ V, jx_sp, atol=1e-12)
        assert np.allclose(V.conj().T @ jy_h.to_matrix() @ V, jy_sp, atol=1e-12)
        assert np.allclose(V.conj().T @ jz_h.to_matrix() @ V, jz_sp, atol=1e-12)


def _hypergeometric_schmidt(n: int, k: int, cut: int) -> np.ndarray:
    """Schmidt probabilities of a Dicke state across the first ``cut`` qubits.

    λ_j = C(cut, j) C(n - cut, k - j) / C(n, k), for every j the binomials allow.
    """
    probs = []
    j_lo = max(0, k - (n - cut))
    j_hi = min(cut, k)
    denom = math.comb(n, k)
    for j in range(j_lo, j_hi + 1):
        lam = math.comb(cut, j) * math.comb(n - cut, k - j) / denom
        if lam > 0.0:
            probs.append(lam)
    return np.array(probs, dtype=float)


def test_dicke_states():
    # Phase 21: Dicke states
    # dicke(3, 1) is the W state: both are the normalized sum of the three weight-1 kets.
    assert np.allclose(dicke(3, 1), w_state(3), atol=1e-12)

    # Cutting after ℓ qubits, the Schmidt probabilities are hypergeometric:
    #   λ_j = C(ℓ, j) C(n−ℓ, k−j) / C(n, k)
    # The entropy built from them equals bipartite_entropy.
    for n in range(2, 9):
        for k in range(n + 1):
            state = dicke(n, k)
            for cut in range(1, n):
                probs = _hypergeometric_schmidt(n, k, cut)
                assert np.isclose(np.sum(probs), 1.0, atol=1e-12)
                entropy = float(-np.sum(probs * np.log2(probs)))
                assert np.isclose(entropy, bipartite_entropy(state, cut), atol=1e-10)


def _dicke_fisher(n: int, k: int) -> float:
    """F = 2 (j(j+1) − m²) with j = n/2 and m = n/2 − k."""
    j = n / 2.0
    m = n / 2.0 - k
    return 2.0 * (j * (j + 1.0) - m * m)


def test_dicke_fisher():
    # Phase 22: Fisher information of Dicke states
    # In |j, m⟩, Var(J_x) = (j(j+1) − m²)/2, so the pure-state QFI is
    # F = 4 Var(J_x) = 2 (j(j+1) − m²).
    for n in range(1, 9):
        jx = collective_spin(n)[0]
        for k in range(n + 1):
            assert np.isclose(qfi(dicke(n, k), jx), _dicke_fisher(n, k), atol=1e-10)

    for n in (16, 32, 64):
        jx = spin_matrices(n / 2.0)[0]
        for k in range(n + 1):
            ket = np.zeros(n + 1, dtype=complex)
            ket[k] = 1.0
            assert np.isclose(qfi(ket, jx), _dicke_fisher(n, k), atol=1e-10)

    # At half filling, F = n(n+2)/2. That beats a product state's n for every
    # n ≥ 1 and falls short of GHZ's n² for every n ≥ 3.
    for n in range(1, 9):
        half = n * (n + 2) / 2.0
        assert np.isclose(half, _dicke_fisher(n, n / 2.0), atol=1e-12)
        assert half > n
        if n >= 3:
            assert half < n * n
        if n % 2 == 0:
            jx, _, jz = collective_spin(n)
            product = PLUS
            for _ in range(n - 1):
                product = np.kron(product, PLUS)
            assert np.isclose(qfi(product, jz), n, atol=1e-10)
            assert np.isclose(qfi(ghz(n), jz), n * n, atol=1e-10)
            assert np.isclose(qfi(dicke(n, n // 2), jx), half, atol=1e-10)


def _product_plus(n: int) -> np.ndarray:
    state = PLUS
    for _ in range(n - 1):
        state = np.kron(state, PLUS)
    return state


def _spin_expectations(state: np.ndarray, ops) -> tuple[float, float, float]:
    return tuple(float(np.real(np.vdot(state, op @ state))) for op in ops)


def test_one_axis_twisting():
    # Phase 23: One-axis twisting
    # H = χ J_z², starting from |+⟩^{⊗n}.
    # Kitagawa and Ueda, Phys. Rev. A 47, 5138 (1993):
    #   ⟨J_x(t)⟩ = (n/2) cos^{n−1}(χ t),   ⟨J_y⟩ = ⟨J_z⟩ = 0.
    chi = 0.1
    times = (0.5, 2.0, 7.0)

    for n in range(1, 9):
        _, _, jz = collective_spin(n)
        h = (chi * (jz @ jz)).simplify()
        psi0 = _product_plus(n)
        jx_m, jy_m, jz_m = (op.to_matrix() for op in collective_spin(n))
        for t in times:
            psi = evolve(h, psi0, t)
            jx, jy, jz_exp = _spin_expectations(psi, (jx_m, jy_m, jz_m))
            expected = (n / 2.0) * math.cos(chi * t) ** (n - 1)
            assert np.isclose(jx, expected, atol=1e-10)
            assert np.isclose(jy, 0.0, atol=1e-10)
            assert np.isclose(jz_exp, 0.0, atol=1e-10)

    for n in (16, 32, 64):
        ops = spin_matrices(n / 2.0)
        for t in times:
            psi = one_axis_state(n, chi, t)
            jx, jy, jz_exp = _spin_expectations(psi, ops)
            expected = (n / 2.0) * math.cos(chi * t) ** (n - 1)
            assert np.isclose(jx, expected, atol=1e-10)
            assert np.isclose(jy, 0.0, atol=1e-10)
            assert np.isclose(jz_exp, 0.0, atol=1e-10)


def test_one_axis_squeezing():
    # Phase 24: Squeezing
    # ξ² = n V_min / ⟨J_x⟩². At t = 0 the state is coherent: V_min = n/4,
    # ⟨J_x⟩ = n/2, so ξ² = 1.
    # Pezzé and Smerzi, Phys. Rev. Lett. 102, 100401 (2009):
    #   F ≥ ⟨J_x⟩² / V_min = n / ξ²,
    # and on a pure state F = 4 Var(J_⊥).
    chi = 0.1
    times = (0.0, 0.5, 1.0, 2.0, 7.0)
    for n in (8, 64):
        for t in times:
            xi2, fisher, bound = one_axis_squeezing(n, chi, t)
            if t == 0.0:
                assert np.isclose(xi2, 1.0, atol=1e-10)
                assert np.isclose(fisher, n, atol=1e-8)
            assert fisher >= bound - 1e-8

    # Kitagawa–Ueda squeezing at n = 64, χt = 0.05.
    xi_squeezed, _, _ = one_axis_squeezing(64, 1.0, 0.05)
    assert xi_squeezed < 1.0
