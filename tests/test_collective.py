import numpy as np
import pytest

from eigenlab.collective import collective_spin, dicke, spin_matrices


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
