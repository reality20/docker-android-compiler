import unittest
import numpy as np
from quantum_engine import QuantumCore, QPU

class TestQuantumSimulation(unittest.TestCase):
    def test_initial_state(self):
        core = QuantumCore(0)

        self.assertAlmostEqual(abs(core.state[0]), 1.0)
        self.assertAlmostEqual(np.sum(np.abs(core.state)**2), 1.0)

    def test_hadamard_gate(self):
        core = QuantumCore(0)
        core.h(0)

        self.assertAlmostEqual(abs(core.state[0])**2, 0.5)
        self.assertAlmostEqual(abs(core.state[1])**2, 0.5)

    def test_bell_state(self):

        core = QuantumCore(0)
        core.h(0)
        core.cnot(0, 1)

        self.assertAlmostEqual(abs(core.state[0])**2, 0.5)
        self.assertAlmostEqual(abs(core.state[3])**2, 0.5)
        self.assertAlmostEqual(abs(core.state[1]), 0.0)
        self.assertAlmostEqual(abs(core.state[2]), 0.0)

    def test_measure(self):
        core = QuantumCore(0)
        core.x(0)
        res = core.measure()

        self.assertTrue(res.endswith('1'))

        self.assertAlmostEqual(abs(core.state[1]), 1.0)

if __name__ == '__main__':
    unittest.main()
