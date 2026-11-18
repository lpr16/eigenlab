import math

import numpy as np
import pytest

from eigenlab.collective import collective_spin, dicke, spin_matrices
from eigenlab.states import bipartite_entropy, w_state


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
