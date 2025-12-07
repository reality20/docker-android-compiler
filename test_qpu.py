import mps_sim
import numpy as np

def test_qpu():
    # Test Bell State
    print("Testing Bell State...")
    qpu = mps_sim.QPU(2, 2)
    qpu.h(0)
    qpu.cx(0, 1)

    # We must flush to execute kernels
    qpu.flush()

    # Reconstruct state from Vidal form
    # State = L[-1] * G[0] * L[0] * G[1] * L[1]
    # L[-1] is boundary 1.0
    # G[0]: (1, 2, 2)
    # L[0]: (2,)
    # G[1]: (2, 2, 1)
    # L[1]: boundary 1.0 (vector of 2, but only [0] matters if boundary)

    # Actually gammas are (N, 2, 2, 2).
    # Bond 0 is Left of Q0.
    # Bond 1 is Between Q0, Q1.
    # Bond 2 is Right of Q1.

    G0 = qpu.gammas[0] # (2, 2, 2)
    L0 = qpu.lambdas[0] # Left of Q0 (Boundary)
    L1 = qpu.lambdas[1] # Between Q0, Q1
    G1 = qpu.gammas[1] # (2, 2, 2)
    L2 = qpu.lambdas[2] # Right of Q1

    # Contract
    # T0(a, i, k) = L0(a) * G0(a, i, k)
    # T0_L1(a, i, k) = T0 * L1(k)
    # T1(k, j, b) = G1(k, j, b) * L2(b)

    # Psi(a, i, j, b) = sum_k T0_L1(a, i, k) * T1(k, j, b)

    # Construct T0_L1
    T0_L1 = np.zeros((2, 2, 2), dtype=np.complex128)
    for a in range(2):
        for i in range(2):
            for k in range(2):
                T0_L1[a, i, k] = L0[a] * G0[a, i, k] * L1[k]

    # Construct T1
    T1 = np.zeros((2, 2, 2), dtype=np.complex128)
    for k in range(2):
        for j in range(2):
            for b in range(2):
                T1[k, j, b] = G1[k, j, b] * L2[b]

    # Contract
    Psi = np.tensordot(T0_L1, T1, axes=(2, 0)) # (a, i, j, b)

    # Boundary conditions: a=0, b=0 usually for |0...0>
    psi_vec = Psi[0, :, :, 0].flatten()

    print("State vector:", psi_vec)

    assert np.isclose(np.abs(psi_vec[0]), 1/np.sqrt(2)) # |00>
    assert np.isclose(np.abs(psi_vec[3]), 1/np.sqrt(2)) # |11>
    assert np.isclose(np.abs(psi_vec[1]), 0)
    assert np.isclose(np.abs(psi_vec[2]), 0)

    print("Bell state verification passed.")

    # Test 3D Topology
    print("Testing 3D Topology...")
    qpu3d = mps_sim.QPU(27, 2) # 3x3x3
    # Neighbors of 0 (0,0,0) should be (1,0,0)->1, (0,1,0)->3, (0,0,1)->9
    # Wait, mapping: z + y*H + x*W*H?
    # Code:
    # z = idx % H
    # y = (idx // H) % W
    # x = idx // (H * W)
    # So 0 is (0,0,0)
    # 1 is (0,0,1) -> Neighbors 0 and 1 (z-diff)
    # 3 is (0,1,0) -> Neighbors 0 and 3 (y-diff)?
    # H=3, W=3.
    # 3 % 3 = 0 (z=0). (3//3)%3 = 1 (y=1). x=0. -> (0,1,0). Yes.

    assert qpu3d._are_neighbors(0, 1)
    assert qpu3d._are_neighbors(0, 3)
    assert qpu3d._are_neighbors(0, 9)
    assert not qpu3d._are_neighbors(0, 2) # (0,0,2) distance 2? No (0,0,0) to (0,0,2) is dist 2.

    print("Topology verification passed.")

if __name__ == "__main__":
    test_qpu()
