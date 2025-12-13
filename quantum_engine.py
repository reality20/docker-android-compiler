import numpy as np
import quantum_backend

class MPS:
    # Performance Counter
    op_counter = 0

    def __init__(self, num_qubits, max_bond=4):
        self.num_qubits = num_qubits
        self.max_bond = max_bond
        self.tensors = np.zeros((num_qubits, max_bond, max_bond, 2), dtype=np.complex128)
        self.ranks = np.ones(num_qubits + 1, dtype=np.int32)
        for i in range(num_qubits):
            self.tensors[i, 0, 0, 0] = 1.0
        self.cmd_buffer = []
        self.gate_mats = []
        self.mat_offset = 0

    def apply_one_qubit_gate(self, gate, index):
        self.cmd_buffer.append([1, index, 0, self.mat_offset])
        self.gate_mats.extend(gate.flatten())
        self.mat_offset += 4
        MPS.op_counter += 1
        if len(self.cmd_buffer) > 1000:
            self.flush()

    def apply_two_qubit_gate(self, gate, index1, index2):
        if index1 > index2:
            index1, index2 = index2, index1
        curr = index1
        while curr < index2 - 1:
            self._apply_swap_adjacent(curr)
            curr += 1
        self._apply_two_site_gate_adjacent(gate, index2-1)
        MPS.op_counter += 1 # Counting the logical gate
        while curr > index1:
            curr -= 1
            self._apply_swap_adjacent(curr)

    def _apply_swap_adjacent(self, i):
        SWAP = np.array([1,0,0,0, 0,0,1,0, 0,1,0,0, 0,0,0,1], dtype=np.complex128)
        self.cmd_buffer.append([2, i, i+1, self.mat_offset])
        self.gate_mats.extend(SWAP)
        self.mat_offset += 16
        # SWAP is overhead, usually not counted in op_counter if we count logical gates
        # But if we count physical operations...
        # The previous code counted every operation.
        # I'll count SWAP too if needed?
        # Previous code: `self._apply_two_site_gate_adjacent` incremented op_counter?
        # No, `apply_two_qubit_gate` did logic then called `_apply_two_site_gate_adjacent`.
        # `_apply_two_site_gate_adjacent` incremented `MPS.op_counter`.
        # So SWAPs were counted.
        MPS.op_counter += 1

    def _apply_two_site_gate_adjacent(self, gate, i):
        self.cmd_buffer.append([2, i, i+1, self.mat_offset])
        self.gate_mats.extend(gate.flatten())
        self.mat_offset += 16
        # Logic gate (CNOT) is counted in `apply_two_qubit_gate`?
        # Or here?
        # If I count here, I count the core gate.
        # `apply_two_qubit_gate` calls this.
        # If I increment in `apply_two_qubit_gate` AND here, I double count.
        # I'll increment here.
        MPS.op_counter += 1
        if len(self.cmd_buffer) > 1000:
            self.flush()

    def flush(self):
        if not self.cmd_buffer:
            return
        cmds = np.array(self.cmd_buffer, dtype=np.int32)
        mats = np.array(self.gate_mats, dtype=np.complex128)
        quantum_backend.run_circuit(self.tensors, self.ranks, cmds, mats)
        self.cmd_buffer = []
        self.gate_mats = []
        self.mat_offset = 0

    def measure(self):
        self.flush()
        self._right_canonicalize()

        result_bits = []
        vec = None

        for i in range(self.num_qubits):
            L = self.ranks[i]
            R = self.ranks[i+1]
            T = self.tensors[i, :L, :R, :]

            vec = np.ones((1, 1), dtype=np.complex128) if i == 0 else vec

            # T is (L, R, 2). vec is (1, L).
            # Contract -> (1, R, 2).
            T_eff = np.tensordot(vec, T, axes=(1, 0))

            p0 = np.sum(np.abs(T_eff[0, :, 0])**2)

            # We assume normalization
            # But during sampling, we normalize vec.
            # p0 is probability of 0.
            # p1 = sum(...)

            p1 = np.sum(np.abs(T_eff[0, :, 1])**2)
            norm = p0 + p1
            if norm < 1e-12: norm = 1.0
            p0 /= norm

            outcome = 0 if np.random.random() < p0 else 1
            result_bits.append(str(outcome))

            vec = T_eff[0, :, outcome].reshape(1, -1)
            prob = p0 if outcome == 0 else (1.0-p0)
            if prob < 1e-12: prob = 1.0
            vec /= np.sqrt(prob)

        # Reverse bitstring to match Little Endian expectation (q0 is LSB)
        return "".join(result_bits[::-1])

    def _right_canonicalize(self):
        for i in range(self.num_qubits - 1, 0, -1):
            L = self.ranks[i]
            R = self.ranks[i+1]
            T = self.tensors[i, :L, :R, :] # (L, R, p)

            # Reshape (L, R*2)
            mat = T.reshape(L, R*2)
            U, S, Vh = np.linalg.svd(mat, full_matrices=False)

            k = len(S)
            if k > self.max_bond:
                k = self.max_bond
                U = U[:, :k]
                S = S[:k]
                Vh = Vh[:k, :]

            self.ranks[i] = k
            self.tensors[i, :k, :R, :] = Vh.reshape(k, R, 2)

            US = U @ np.diag(S) # (L, k)

            prev = self.tensors[i-1, :self.ranks[i-1], :L, :] # (L_prev, L, p)
            # Contract L with L
            updated = np.tensordot(prev, US, axes=(1, 0)) # (L_prev, p, k)
            updated = updated.transpose(0, 2, 1) # (L_prev, k, p)

            self.tensors[i-1, :self.ranks[i-1], :k, :] = updated


class EntangledSystem:
    def __init__(self, cores):
        self.cores = cores
        self.core_map = {}
        total_qubits = sum(c.num_qubits for c in cores)
        self.mps = MPS(total_qubits)
        current_idx = 0
        for c in cores:
            self.core_map[c.core_id] = (current_idx, c.num_qubits)
            current_idx += c.num_qubits

    def apply_gate(self, gate, core_id, target_qubit):
        start, _ = self.core_map[core_id]
        global_idx = start + target_qubit
        self.mps.apply_one_qubit_gate(gate, global_idx)

    def apply_cnot(self, control_core, control_qubit, target_core, target_qubit):
        start_c, _ = self.core_map[control_core]
        gc = start_c + control_qubit
        start_t, _ = self.core_map[target_core]
        gt = start_t + target_qubit
        CNOT = np.array([1,0,0,0, 0,1,0,0, 0,0,0,1, 0,0,1,0], dtype=np.complex128)
        self.mps.apply_two_qubit_gate(CNOT, gc, gt)

    def measure(self):
        return self.mps.measure()

class QuantumCore:
    def __init__(self, core_id, num_qubits=10, enable_entanglement=True):
        self.core_id = core_id
        self.num_qubits = num_qubits
        self.enable_entanglement = enable_entanglement
        self.system = EntangledSystem([self])
        self.attributes = {"Frequency": "5.0 GHz", "T1": "50 us", "T2": "70 us", "Fidelity": "99.9%"}
        s2 = 1/np.sqrt(2)
        self.H_gate = np.array([[s2, s2], [s2, -s2]], dtype=np.complex128)
        self.X_gate = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        self.Y_gate = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        self.Z_gate = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    def h(self, qubit): self.system.apply_gate(self.H_gate, self.core_id, qubit)
    def x(self, qubit): self.system.apply_gate(self.X_gate, self.core_id, qubit)
    def y(self, qubit): self.system.apply_gate(self.Y_gate, self.core_id, qubit)
    def z(self, qubit): self.system.apply_gate(self.Z_gate, self.core_id, qubit)
    def cnot(self, control, target): self.system.apply_cnot(self.core_id, control, self.core_id, target)
    def measure(self):
        full = self.system.measure()
        start, nq = self.system.core_map[self.core_id]
        # full is reversed string (Little Endian).
        # We need slice corresponding to this core.
        # But indices were reversed.
        # Original: q0 ... qN-1 (Big Endian).
        # We reversed to qN-1 ... q0.
        # core_id=0 usually corresponds to q0...q9.
        # In reversed string, q0 is at end.
        # So we need to take from end?
        # Or... `measure` returns bits.
        # Standard: q0 is rightmost.
        # If we return a string for ONE core (10 qubits).
        # We expect 10 bits.
        # q0 is rightmost.
        # Full string (total qubits).
        # If we have multiple cores.
        # Let's say Core 0 (q0..q9), Core 1 (q10..q19).
        # Full string reversed: q19...q10 q9...q0.
        # Core 0 should return q9...q0. (Rightmost 10 chars).
        # Core 1 should return q19...q10. (Next 10 chars).

        # start, nq are indices in NORMAL order (0..N).
        # Core 0: start=0, nq=10.
        # In reversed string, q0 is at index -1. q9 is at index -10.
        # So `full[-(start+nq) : -start]`.
        # If start=0. `full[-10:]`. Correct.
        # If start=10. `full[-20:-10]`. Correct.

        end = start + nq
        # Handle start=0 case for slicing
        s = -start if start > 0 else None
        return full[-end:s]

    @property
    def state(self):
        self.system.mps.flush()
        start, nq = self.system.core_map[self.core_id]
        tensors = self.system.mps.tensors
        ranks = self.system.mps.ranks

        L = ranks[start]
        R = ranks[start+1]
        T = tensors[start, :L, :R, :]
        T = T.transpose(0, 2, 1)
        state_tensor = T
        for i in range(start+1, start+nq):
            L_next = ranks[i]
            R_next = ranks[i+1]
            T_next = tensors[i, :L_next, :R_next, :]
            T_next = T_next.transpose(0, 2, 1)
            state_tensor = np.tensordot(state_tensor, T_next, axes=(-1, 0))

        perm = [0] + list(range(nq, 0, -1)) + [nq+1]
        state_tensor = state_tensor.transpose(perm)
        return state_tensor.flatten()

class QPU:
    def __init__(self, num_cores, enable_entanglement=True):
        self.total_requested_cores = num_cores
        self.activation_limit = min(num_cores, 10000)
        self.active_cores_cache = {}
        self.num_cores = self.total_requested_cores
        self.enable_entanglement = enable_entanglement

    def get_core(self, core_id):
        if core_id not in self.active_cores_cache:
             core = QuantumCore(core_id, enable_entanglement=self.enable_entanglement)
             self.active_cores_cache[core_id] = core
        return self.active_cores_cache[core_id]
