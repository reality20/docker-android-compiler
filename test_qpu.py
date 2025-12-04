import mps_sim
import numpy as np

def test_qpu():
    # Test Bell State
    qpu = mps_sim.QPU(2, 2)
    qpu.h(0)
    qpu.cx(0, 1)

    # Contract to full state vector to verify
    # Tensors: [2, 1], [2, 1] (roughly, dimensions vary due to truncation)
    # Tensor 0: (1, 2, k)
    # Tensor 1: (k, 2, 1)

    t0 = qpu.tensors[0] # (1, 2, k)
    t1 = qpu.tensors[1] # (k, 2, 1)

    # Contract bond
    psi = np.tensordot(t0, t1, axes=(2, 0)) # (1, 2, 2, 1)
    psi = psi.flatten()

    # Expected: 1/sqrt(2) (|00> + |11>)
    # 00 -> index 0
    # 01 -> index 1
    # 10 -> index 2
    # 11 -> index 3

    print("State vector:", psi)

    assert np.isclose(abs(psi[0]), 1/np.sqrt(2))
    assert np.isclose(abs(psi[3]), 1/np.sqrt(2))
    assert np.isclose(abs(psi[1]), 0)
    assert np.isclose(abs(psi[2]), 0)

    print("Bell state verification passed.")

if __name__ == "__main__":
    test_qpu()
