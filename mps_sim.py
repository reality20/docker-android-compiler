import numpy as np
import kernels
from kernels import (
    GATE_I, GATE_X, GATE_Y, GATE_Z, GATE_H, GATE_S, GATE_T,
    GATE_RX, GATE_RY, GATE_RZ, GATE_CX, GATE_CZ, GATE_SWAP,
    GATE_MEASURE, GATE_RESET
)

class QPU:
    def __init__(self, num_qubits, bond_dim=2, state='zeros'):
        self.num_qubits = num_qubits
        self.bond_dim = 2 # Fixed to 2 for optimization
        self.rng = np.random.default_rng()

        # 3D Topology Dimensions
        # We define a cubic bounding box
        s = int(np.ceil(num_qubits**(1/3)))
        if s == 0: s = 1
        self.L = s
        self.W = s
        self.H = s

        # Vidal Form State
        # Gammas: (N, 2, 2, 2)
        # Lambdas: (N+1, 2)
        self.gammas = np.zeros((num_qubits, 2, 2, 2), dtype=np.complex128)
        self.lambdas = np.zeros((num_qubits + 1, 2), dtype=np.float64)

        # Initialize to |0...0>
        # G[i, 0, 0, 0] = 1.0
        # L[i] = [1.0, 0.0]
        self.gammas[:, 0, 0, 0] = 1.0
        self.lambdas[:, 0] = 1.0

        # Swap Network Mappings
        self.ptr = np.arange(num_qubits, dtype=np.int32) # Logical q -> Physical index
        self.qubits = np.arange(num_qubits, dtype=np.int32) # Physical index -> Logical q

        # Command Buffer
        # Each instruction: [OpCode, PhysQ1, PhysQ2, ParamIdx]
        self.instr_list = []
        self.param_list = []
        self.gate_count = 0

    def _get_coords(self, logical_idx):
        # Map logical index to x,y,z
        H = self.H
        W = self.W
        z = logical_idx % H
        y = (logical_idx // H) % W
        x = logical_idx // (H * W)
        return x, y, z

    def _are_neighbors(self, q1, q2):
        # Check 3D adjacency
        x1, y1, z1 = self._get_coords(q1)
        x2, y2, z2 = self._get_coords(q2)
        dist = abs(x1-x2) + abs(y1-y2) + abs(z1-z2)
        return dist == 1

    def flush(self):
        if not self.instr_list:
            return

        instrs = np.array(self.instr_list, dtype=np.int32)
        params = np.array(self.param_list, dtype=np.float64)

        kernels.run_simulation_kernel(self.gammas, self.lambdas, instrs, params)

        self.instr_list = []
        self.param_list = []

    def _route(self, p1, p2):
        # Route physical qubits p1 and p2 to be adjacent
        if p1 == p2:
            return p1, p2

        # We want to bring them adjacent.
        # Determine direction
        if p1 > p2:
            p1, p2 = p2, p1

        # Now p1 < p2
        # Move p2 down to p1+1
        while p2 > p1 + 1:
            sw_left = p2 - 1
            sw_right = p2

            # Emit SWAP
            self.instr_list.append([GATE_SWAP, sw_left, sw_right, -1])
            self.gate_count += 1

            # Update Map
            l_left = self.qubits[sw_left]
            l_right = self.qubits[sw_right]

            self.qubits[sw_left] = l_right
            self.qubits[sw_right] = l_left
            self.ptr[l_left] = sw_right
            self.ptr[l_right] = sw_left

            p2 -= 1

        return p1, p2

    def apply_1q(self, op, q, param=None):
        if q < 0 or q >= self.num_qubits:
            raise ValueError(f"Qubit {q} out of bounds")

        p = self.ptr[q]

        pidx = -1
        if param is not None:
            pidx = len(self.param_list)
            self.param_list.append(param)

        self.instr_list.append([op, p, -1, pidx])
        self.gate_count += 1

    def apply_2q(self, op, q1, q2, param=None):
        if q1 < 0 or q1 >= self.num_qubits or q2 < 0 or q2 >= self.num_qubits:
            raise ValueError("Qubit index out of bounds")

        # Check Topology
        if not self._are_neighbors(q1, q2):
            raise ValueError(f"Qubits {q1} and {q2} are not neighbors in 3D Hypercube topology")

        p1 = self.ptr[q1]
        p2 = self.ptr[q2]

        # Route
        p1, p2 = self._route(p1, p2)

        # Apply Gate
        # p1 is left, p2 is right (p2 = p1+1)
        pidx = -1
        if param is not None:
            pidx = len(self.param_list)
            self.param_list.append(param)

        self.instr_list.append([op, p1, p2, pidx])
        self.gate_count += 1

    def measure(self, q):
        self.flush()

        p = self.ptr[q]

        # Calculate prob(0) locally
        # T = diag(L[p]) * G[p] * diag(L[p+1])
        # We need to construct this tensor.
        # But we only need T[:, 0, :] and T[:, 1, :] norm.

        L_left = self.lambdas[p]
        G = self.gammas[p]
        L_right = self.lambdas[p+1]

        # Construct T
        # T(a, i, b) = L_left(a) * G(a, i, b) * L_right(b)

        p0 = 0.0
        p1 = 0.0

        # We iterate to sum |T|^2
        # Since bond dim is 2, it's fast.
        for a in range(2):
            for b in range(2):
                # i=0
                val0 = L_left[a] * G[a, 0, b] * L_right[b]
                p0 += np.abs(val0)**2

                # i=1
                val1 = L_left[a] * G[a, 1, b] * L_right[b]
                p1 += np.abs(val1)**2

        # Normalize
        total = p0 + p1
        if total < 1e-15:
             # Should not happen
             p0 = 0.5
        else:
             p0 /= total

        outcome = 0
        if self.rng.random() > p0:
            outcome = 1

        # Collapse State
        # If 0, zero out i=1. Re-normalize.
        # We modify G in place.

        if outcome == 0:
            G[:, 1, :] = 0.0
            norm = np.sqrt(p0)
        else:
            G[:, 0, :] = 0.0
            norm = np.sqrt(1.0 - p0)

        if norm > 1e-15:
            G /= norm

        # Since we modified G in python, we don't need to push to kernel.
        # The kernel reads 'gammas' array which is shared memory (numpy array).
        # Wait, if kernel was running, it would be an issue. But we flushed.

        return outcome

    def reset(self, q):
        m = self.measure(q)
        if m == 1:
            self.x(q)

    def reset_all(self):
        # Reset everything
        self.flush() # Ensure any pending are done (though we clear them)
        self.instr_list = []
        self.param_list = []

        self.gammas.fill(0.0)
        self.gammas[:, 0, 0, 0] = 1.0

        self.lambdas.fill(0.0)
        self.lambdas[:, 0] = 1.0

        self.ptr = np.arange(self.num_qubits, dtype=np.int32)
        self.qubits = np.arange(self.num_qubits, dtype=np.int32)
        self.gate_count = 0

    def memory_usage(self):
        return self.gammas.nbytes + self.lambdas.nbytes + \
               (len(self.instr_list) * 16) + (len(self.param_list) * 8)

    # Gate Wrappers
    def h(self, q): self.apply_1q(GATE_H, q)
    def s(self, q): self.apply_1q(GATE_S, q)
    def t(self, q): self.apply_1q(GATE_T, q)
    def x(self, q): self.apply_1q(GATE_X, q)
    def y(self, q): self.apply_1q(GATE_Y, q)
    def z(self, q): self.apply_1q(GATE_Z, q)
    def rx(self, q, theta): self.apply_1q(GATE_RX, q, theta)
    def ry(self, q, theta): self.apply_1q(GATE_RY, q, theta)
    def rz(self, q, theta): self.apply_1q(GATE_RZ, q, theta)

    def cx(self, c, t): self.apply_2q(GATE_CX, c, t)
    def cz(self, c, t): self.apply_2q(GATE_CZ, c, t)
    def swap(self, c, t): self.apply_2q(GATE_SWAP, c, t)

