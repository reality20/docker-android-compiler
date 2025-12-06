import mps_sim
import numpy as np

def test_qpu():
    # Test Bell State
    qpu = mps_sim.QPU(2, 2)
    qpu.h(0)
    qpu.cx(0, 1)
    qpu.flush()

    # Reconstruct state from Gamma-Lambda form
    # L0 G0 L1 G1 L2

    l0 = qpu.lambdas[0]
    g0 = qpu.gammas[0]
    l1 = qpu.lambdas[1]
    g1 = qpu.gammas[1]
    l2 = qpu.lambdas[2]

    # Contract: L0 * G0 * L1 * G1 * L2
    # L0, L1, L2 are vectors (diagonal matrices)

    # Tensor 0: L0 * G0 * sqrt(L1) ? No, symmetric form usually.
    # Standard: Psi = L0 G0 L1 G1 L2.

    # Contract L0 * G0 -> T1
    # T1[a, i, b] = L0[a] * G0[a, i, b]
    t1 = np.zeros_like(g0)
    for a in range(2):
        for i in range(2):
            for b in range(2):
                t1[a, i, b] = l0[a] * g0[a, i, b]

    # Contract T1 * L1 -> T2
    # T2[a, i, b] = T1[a, i, b] * L1[b]
    t2 = np.zeros_like(t1)
    for a in range(2):
        for i in range(2):
            for b in range(2):
                t2[a, i, b] = t1[a, i, b] * l1[b]

    # Contract T2 * G1 -> T3 (merge bond b)
    # T3[a, i, j, c] = sum_b T2[a, i, b] * G1[b, j, c]
    t3 = np.tensordot(t2, g1, axes=(2, 0)) # (2, 2, 2, 2)

    # Contract T3 * L2
    # T4[a, i, j, c] = T3[a, i, j, c] * L2[c]
    t4 = np.zeros_like(t3)
    for a in range(2):
        for i in range(2):
            for j in range(2):
                for c in range(2):
                    t4[a, i, j, c] = t3[a, i, j, c] * l2[c]

    # Flatten
    # Since boundary L0=[1,0], L2=[1,0], we only care about indices a=0, c=0.
    psi = t4[0, :, :, 0].flatten()

    # Expected: 1/sqrt(2) (|00> + |11>)
    print("State vector:", psi)

    assert np.isclose(abs(psi[0]), 1/np.sqrt(2))
    assert np.isclose(abs(psi[3]), 1/np.sqrt(2))
    assert np.isclose(abs(psi[1]), 0)
    assert np.isclose(abs(psi[2]), 0)

    print("Bell state verification passed.")

if __name__ == "__main__":
    test_qpu()
