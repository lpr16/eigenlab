import numpy as np

from eigenlab.fermions import (
    jordan_wigner,
    jordan_wigner_a,
    jordan_wigner_adag,
    number_operator,
    total_number_operator,
)
from eigenlab.hamiltonian import to_matrix
from eigenlab.states import basis


def test_jordan_wigner_anticommutation_relations():
    # Phase 14:
    # Map a_p and a†_p to Pauli strings with the frozen qubit order.
    # For n ≤ 4:
    # 1. {a_i, a†_j} = δ_ij I
    # 2. {a_i, a_j} = 0
    # 3. a_i² = 0
    # 4. The vacuum is basis(0, n)
    # 5. a†_p |0⟩ is a single computational-basis vector.

    for n in [1, 2, 3, 4]:
        dim = 1 << n
        identity = np.eye(dim, dtype=complex)
        zero = np.zeros((dim, dim), dtype=complex)
        vac = basis(0, n)

        # Precompute matrix representations for all orbitals
        a_mats = [jordan_wigner_a(p, n).to_matrix() for p in range(n)]
        adag_mats = [jordan_wigner_adag(p, n).to_matrix() for p in range(n)]

        for i in range(n):
            ai = a_mats[i]
            ai_dag = adag_mats[i]

            # 3. a_i² = 0
            assert np.allclose(ai @ ai, zero)
            assert np.allclose(ai_dag @ ai_dag, zero)

            # Check via term operator multiplication as well
            ai_h = jordan_wigner_a(i, n)
            ai_dag_h = jordan_wigner_adag(i, n)
            assert np.allclose((ai_h @ ai_h).to_matrix(), zero)

            # 5. a†_p |0⟩ is a single computational-basis vector:
            # Qubit order contract: qubit 0 is most significant bit.
            # Orbital p occupied => bit p is 1 => index is 1 << (n - 1 - p).
            created_ket = ai_dag @ vac
            expected_index = 1 << (n - 1 - i)
            expected_ket = basis(expected_index, n)
            assert np.allclose(created_ket, expected_ket)

            # Annihilation on vacuum gives 0
            assert np.allclose(ai @ vac, np.zeros(dim, dtype=complex))

            for j in range(n):
                aj = a_mats[j]
                aj_dag = adag_mats[j]

                # 1. {a_i, a†_j} = δ_ij I
                anticom_adag = ai @ aj_dag + aj_dag @ ai
                expected_adag = identity if i == j else zero
                assert np.allclose(anticom_adag, expected_adag)

                # Check via term algebra:
                aj_dag_h = jordan_wigner_adag(j, n)
                term_anticom = (ai_h @ aj_dag_h) + (aj_dag_h @ ai_h)
                assert np.allclose(term_anticom.to_matrix(), expected_adag)

                # 2. {a_i, a_j} = 0
                anticom_aa = ai @ aj + aj @ ai
                assert np.allclose(anticom_aa, zero)


def test_jordan_wigner_number_operators():
    # n_p = a†_p a_p = (I - Z_p)/2
    for n in [1, 2, 3]:
        for p in range(n):
            a_p = jordan_wigner_a(p, n)
            adag_p = jordan_wigner_adag(p, n)
            n_op = number_operator(p, n)
            assert np.allclose((adag_p @ a_p).to_matrix(), n_op.to_matrix())
