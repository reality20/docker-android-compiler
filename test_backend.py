import unittest
import numpy as np
import quantum_backend

class TestQuantumBackend(unittest.TestCase):

    def test_one_qubit_gate(self):
        # New Layout: (N, max_bond, max_bond, 2)
        num_qubits = 1
        max_bond = 4
        tensors = np.zeros((num_qubits, max_bond, max_bond, 2), dtype=np.complex128)
        ranks = np.ones(num_qubits + 1, dtype=np.int32)

        # Initialize |0>
        # l=0, r=0, p=0
        tensors[0, 0, 0, 0] = 1.0

        # Gate X
        X = np.array([0, 1, 1, 0], dtype=np.complex128)

        # Command
        cmds = np.array([[1, 0, 0, 0]], dtype=np.int32)
        mats = X

        quantum_backend.run_circuit(tensors, ranks, cmds, mats)

        # Check result: |1> -> p=1 should be 1.0
        # Index: [0, 0, 0, 1]
        self.assertAlmostEqual(abs(tensors[0, 0, 0, 1]), 1.0)
        self.assertAlmostEqual(abs(tensors[0, 0, 0, 0]), 0.0)

    def test_two_qubit_gate_logic(self):
        # New Layout: (N, max_bond, max_bond, 2)
        num_qubits = 2
        max_bond = 4
        tensors = np.zeros((num_qubits, max_bond, max_bond, 2), dtype=np.complex128)
        ranks = np.ones(num_qubits + 1, dtype=np.int32)

        tensors[0, 0, 0, 0] = 1.0
        tensors[1, 0, 0, 0] = 1.0

        # Apply H on 0
        H = np.array([1, 1, 1, -1], dtype=np.complex128) / np.sqrt(2)
        CNOT = np.array([1,0,0,0, 0,1,0,0, 0,0,0,1, 0,0,1,0], dtype=np.complex128)

        cmds = np.array([
            [1, 0, 0, 0], # H on 0
            [2, 0, 1, 4]  # CNOT on 0,1
        ], dtype=np.int32)

        mats = np.concatenate((H, CNOT))

        quantum_backend.run_circuit(tensors, ranks, cmds, mats)

        # Check Bell State
        r = ranks[1]

        # T0: (1, r, 2) -> (1, 2, r)
        T0 = tensors[0, :1, :r, :].transpose(0, 2, 1)
        # T1: (r, 1, 2) -> (r, 2, 1)
        T1 = tensors[1, :r, :1, :].transpose(0, 2, 1)

        # Contract
        # T0 (1, 2, r) dot T1 (r, 2, 1) -> (1, 2, 2, 1)
        psi = np.tensordot(T0, T1, axes=(2, 0))
        psi = psi.flatten()

        # |00> + |11>
        # p0=0, p1=0 (idx 0)
        # p0=0, p1=1 (idx 1)
        # p0=1, p1=0 (idx 2)
        # p0=1, p1=1 (idx 3)

        self.assertAlmostEqual(abs(psi[0])**2, 0.5)
        self.assertAlmostEqual(abs(psi[3])**2, 0.5)
        self.assertAlmostEqual(abs(psi[1]), 0.0)
        self.assertAlmostEqual(abs(psi[2]), 0.0)

if __name__ == '__main__':
    unittest.main()
