import numpy as np
from numba import njit, prange, int32, float64, complex128

# Constants
OP_I, OP_X, OP_Y, OP_Z = 0, 1, 2, 3
OP_H, OP_S, OP_T = 4, 5, 6
OP_RX, OP_RY, OP_RZ = 7, 8, 9
OP_CX, OP_CZ, OP_SWAP = 10, 11, 12
OP_MEASURE, OP_RESET = 13, 14

# --- Precomputed Gates ---
GATES_1Q = np.zeros((10, 2, 2), dtype=np.complex128)
GATES_1Q[OP_I] = np.eye(2, dtype=np.complex128)
GATES_1Q[OP_X] = np.array([[0, 1], [1, 0]], dtype=np.complex128)
GATES_1Q[OP_Y] = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
GATES_1Q[OP_Z] = np.array([[1, 0], [0, -1]], dtype=np.complex128)
GATES_1Q[OP_H] = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
GATES_1Q[OP_S] = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
GATES_1Q[OP_T] = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=np.complex128)

GATES_2Q = np.zeros((13, 4, 4), dtype=np.complex128)
GATES_2Q[OP_CX] = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=np.complex128)
GATES_2Q[OP_CZ] = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]], dtype=np.complex128)
GATES_2Q[OP_SWAP] = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=np.complex128)

@njit(fastmath=True, cache=True)
def get_parameterized_gate(opcode, theta):
    res = np.zeros((2, 2), dtype=np.complex128)
    if opcode == OP_RX:
        c = np.cos(theta/2); s = -1j * np.sin(theta/2)
        res[0,0]=c; res[0,1]=s; res[1,0]=s; res[1,1]=c
    elif opcode == OP_RY:
        c = np.cos(theta/2); s = -np.sin(theta/2)
        res[0,0]=c; res[0,1]=s; res[1,0]=-s; res[1,1]=c
    elif opcode == OP_RZ:
        res[0,0]=np.exp(-1j*theta/2); res[1,1]=np.exp(1j*theta/2)
    return res

# --- Linear Algebra ---
@njit(fastmath=True, cache=True)
def fast_svd_4x4_vals_vecs(A):
    H = A @ A.T.conj()
    V = np.eye(4, dtype=np.complex128)
    curr_A = H

    # 3 Sweeps
    for _ in range(3):
        pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
        for p, q in pairs:
            app = curr_A[p,p].real; aqq = curr_A[q,q].real; apq = curr_A[p,q]
            mag = abs(apq)
            if mag < 1e-15: continue

            b11 = app; b22 = aqq; b12 = apq
            tr = b11 + b22; diff = b11 - b22
            delta = np.sqrt(diff*diff + 4*b12*np.conj(b12))
            l1 = (tr + delta)/2; l2 = (tr - delta)/2

            v1_0 = b12; v1_1 = l1 - b11
            n1 = np.sqrt(abs(v1_0)**2 + abs(v1_1)**2)
            if n1==0: n1=1.0
            v1_0/=n1; v1_1/=n1

            v2_0 = b12; v2_1 = l2 - b11
            n2 = np.sqrt(abs(v2_0)**2 + abs(v2_1)**2)
            if n2==0: n2=1.0
            v2_0/=n2; v2_1/=n2

            for i in range(4):
                vp = V[i,p]; vq = V[i,q]
                V[i,p] = vp*v1_0 + vq*v1_1
                V[i,q] = vp*v2_0 + vq*v2_1

            J = np.eye(4, dtype=np.complex128)
            J[p,p]=v1_0; J[q,p]=v1_1; J[p,q]=v2_0; J[q,q]=v2_1
            curr_A = J.conj().T @ curr_A @ J

    evals = np.diag(curr_A).real
    i1, i2 = 0, 1
    v1, v2 = evals[0], evals[1]
    if v2 > v1: i1, i2 = 1, 0; v1, v2 = v2, v1
    for k in range(2, 4):
        val = evals[k]
        if val > v1: i2=i1; v2=v1; i1=k; v1=val
        elif val > v2: i2=k; v2=val

    s1 = np.sqrt(max(0.0, v1)); s2 = np.sqrt(max(0.0, v2))

    U = np.zeros((4, 2), dtype=np.complex128)
    U[:, 0] = V[:, i1]; U[:, 1] = V[:, i2]

    S = np.zeros(2, dtype=np.float64)
    S[0] = s1; S[1] = s2

    Vh = np.zeros((2, 4), dtype=np.complex128)
    if s1 > 1e-15: Vh[0,:] = (U[:,0].conj() @ A) / s1
    if s2 > 1e-15: Vh[1,:] = (U[:,1].conj() @ A) / s2

    return U, S, Vh

@njit(fastmath=True, cache=True)
def measure_qubit(gammas, lambdas, q, rand_val):
    # Contract local state
    # L[q] * G[q] * L[q+1]
    l1 = lambdas[q]
    g = gammas[q]
    l2 = lambdas[q+1]

    # Psi (2, 2, 2)
    psi = np.zeros((2, 2, 2), dtype=np.complex128)
    for a in range(2):
        for i in range(2):
            for b in range(2):
                psi[a, i, b] = l1[a] * g[a, i, b] * l2[b]

    # Prob 0
    p0 = 0.0
    for a in range(2):
        for b in range(2):
            val = psi[a, 0, b]
            p0 += (val.real**2 + val.imag**2)

    outcome = 0
    if rand_val > p0:
        outcome = 1

    # Collapse
    norm = 0.0
    if outcome == 0:
        psi[:, 1, :] = 0.0
        norm = np.sqrt(p0)
    else:
        psi[:, 0, :] = 0.0
        norm = np.sqrt(1.0 - p0)

    if norm < 1e-15: norm = 1.0
    psi /= norm

    # Restore Vidal form (SVD)
    # Psi(a, i, b) -> Matrix M(a*i, b) ? No.
    # We want G' such that L1 G' L2 = Psi'.
    # So G' = L1^-1 Psi' L2^-1.

    new_g = np.zeros((2, 2, 2), dtype=np.complex128)
    for a in range(2):
        inv1 = 1.0/l1[a] if abs(l1[a]) > 1e-12 else 0.0
        for b in range(2):
            inv2 = 1.0/l2[b] if abs(l2[b]) > 1e-12 else 0.0
            for i in range(2):
                new_g[a, i, b] = psi[a, i, b] * inv1 * inv2

    gammas[q] = new_g
    return outcome

@njit(fastmath=True, cache=True)
def apply_gate_kernel(gammas, lambdas, op, q1, q2, param, measure_out, rand_vals, meas_ptr):
    if op == OP_MEASURE:
        outcome = measure_qubit(gammas, lambdas, q1, rand_vals[meas_ptr])
        measure_out[q1] = outcome

    elif op == OP_RESET:
        # Measure, if 1 apply X.
        outcome = measure_qubit(gammas, lambdas, q1, rand_vals[meas_ptr])
        if outcome == 1:
            # Apply X
            old_g = gammas[q1]
            new_g = np.zeros((2, 2, 2), dtype=np.complex128)
            g = GATES_1Q[OP_X]
            for a in range(2):
                for b in range(2):
                    new_g[a, 0, b] = old_g[a, 0, b]*g[0,0] + old_g[a, 1, b]*g[0,1]
                    new_g[a, 1, b] = old_g[a, 0, b]*g[1,0] + old_g[a, 1, b]*g[1,1]
            gammas[q1] = new_g

    elif op >= 10: # 2Q
        g1 = gammas[q1]; l1 = lambdas[q1]; g2 = gammas[q2]; l2 = lambdas[q1+1]; l3 = lambdas[q2+1]
        gate = GATES_2Q[op]

        theta = np.zeros((2, 4, 2), dtype=np.complex128)
        for a in range(2):
            val_l1 = l1[a]
            for c in range(2):
                val_l3 = l3[c]
                for i in range(2):
                    for j in range(2):
                        acc = 0j
                        for b in range(2): acc += g1[a, i, b] * l2[b] * g2[b, j, c]
                        theta[a, i*2+j, c] = acc * val_l1 * val_l3

        theta_prime = np.zeros((2, 4, 2), dtype=np.complex128)
        for a in range(2):
            for c in range(2):
                for r in range(4):
                    acc = 0j
                    for k in range(4): acc += theta[a, k, c] * gate[r, k]
                    theta_prime[a, r, c] = acc

        M = np.zeros((4, 4), dtype=np.complex128)
        for a in range(2):
            for c in range(2):
                for r in range(4): M[a*2+(r//2), (r%2)*2+c] = theta_prime[a, r, c]

        U, S, Vh = fast_svd_4x4_vals_vecs(M)
        lambdas[q1+1] = S / np.linalg.norm(S)

        for a in range(2):
            inv = 1.0/l1[a] if abs(l1[a]) > 1e-12 else 0.0
            for i in range(2):
                for b in range(2): gammas[q1][a, i, b] = U[a*2+i, b] * inv
        for c in range(2):
            inv = 1.0/l3[c] if abs(l3[c]) > 1e-12 else 0.0
            for j in range(2):
                for b in range(2): gammas[q2][b, j, c] = Vh[b, j*2+c] * inv

    else: # 1Q
        if op in [OP_RX, OP_RY, OP_RZ]: g = get_parameterized_gate(op, param)
        else: g = GATES_1Q[op]
        old_g = gammas[q1]; new_g = np.zeros((2, 2, 2), dtype=np.complex128)
        for a in range(2):
            for b in range(2):
                new_g[a, 0, b] = old_g[a, 0, b]*g[0,0] + old_g[a, 1, b]*g[0,1]
                new_g[a, 1, b] = old_g[a, 0, b]*g[1,0] + old_g[a, 1, b]*g[1,1]
        gammas[q1] = new_g

@njit(parallel=True, fastmath=True, cache=True)
def run_scheduler_kernel(gammas, lambdas, gate_queue, gate_params, queue_len, num_qubits, rand_vals, measure_out):
    layer_ids = np.zeros(queue_len, dtype=np.int32)
    qubit_layer = np.zeros(num_qubits, dtype=np.int32)
    max_layer = 0

    for k in range(queue_len):
        op = gate_queue[k, 0]
        q1 = gate_queue[k, 1]
        l = qubit_layer[q1]
        if op >= 10:
            q2 = gate_queue[k, 2]
            l2 = qubit_layer[q2]
            if l2 > l: l = l2
        layer_ids[k] = l
        qubit_layer[q1] = l + 1
        if op >= 10: qubit_layer[gate_queue[k, 2]] = l + 1
        if l > max_layer: max_layer = l

    num_layers = max_layer + 1
    layer_counts = np.zeros(num_layers, dtype=np.int32)
    for k in range(queue_len): layer_counts[layer_ids[k]] += 1

    sorted_indices = np.zeros(queue_len, dtype=np.int32)
    layer_starts = np.zeros(num_layers + 1, dtype=np.int32)
    acc = 0
    for i in range(num_layers): layer_starts[i] = acc; acc += layer_counts[i]
    layer_starts[num_layers] = acc

    current_ptr = np.zeros(num_layers, dtype=np.int32)
    for i in range(num_layers): current_ptr[i] = layer_starts[i]

    for k in range(queue_len):
        lid = layer_ids[k]
        pos = current_ptr[lid]
        sorted_indices[pos] = k
        current_ptr[lid] += 1

    for l in range(num_layers):
        start = layer_starts[l]
        end = layer_starts[l+1]
        for i in prange(start, end):
            k = sorted_indices[i]
            op = gate_queue[k, 0]
            q1 = gate_queue[k, 1]
            q2 = gate_queue[k, 2]
            pidx = gate_queue[k, 3]
            param = 0.0
            if pidx >= 0: param = gate_params[pidx]
            apply_gate_kernel(gammas, lambdas, op, q1, q2, param, measure_out, rand_vals, k)

class QPU:
    def __init__(self, num_qubits, bond_dim=2, state='zeros'):
        self.num_qubits = num_qubits
        self.rng = np.random.default_rng()
        self.gammas = np.zeros((num_qubits, 2, 2, 2), dtype=np.complex128)
        self.lambdas = np.zeros((num_qubits + 1, 2), dtype=np.float64)
        for i in range(num_qubits): self.gammas[i,0,0,0]=1.0
        for i in range(num_qubits+1): self.lambdas[i,0]=1.0

        self.queue_cap = 2000000
        self.gate_queue = np.zeros((self.queue_cap, 4), dtype=np.int32)
        self.gate_params = np.zeros(self.queue_cap, dtype=np.float64)
        self.q_ptr = 0
        self.p_ptr = 0
        self.measure_out = np.zeros(num_qubits, dtype=np.int32)

    def flush(self):
        if self.q_ptr == 0: return
        rand_vals = self.rng.random(self.q_ptr)
        run_scheduler_kernel(self.gammas, self.lambdas, self.gate_queue, self.gate_params, self.q_ptr, self.num_qubits, rand_vals, self.measure_out)
        self.q_ptr = 0
        self.p_ptr = 0

    def _add(self, op, q1, q2=0, p=0.0):
        if self.q_ptr >= self.queue_cap: self.flush()
        idx = self.q_ptr
        self.gate_queue[idx,0]=op; self.gate_queue[idx,1]=q1; self.gate_queue[idx,2]=q2
        pidx = -1
        if op in [OP_RX, OP_RY, OP_RZ]:
            pidx = self.p_ptr
            self.gate_params[pidx] = p
            self.p_ptr += 1
        self.gate_queue[idx,3]=pidx
        self.q_ptr += 1

    def x(self, q): self._add(OP_X, q)
    def y(self, q): self._add(OP_Y, q)
    def z(self, q): self._add(OP_Z, q)
    def h(self, q): self._add(OP_H, q)
    def s(self, q): self._add(OP_S, q)
    def t(self, q): self._add(OP_T, q)
    def rx(self, q, th): self._add(OP_RX, q, p=th)
    def ry(self, q, th): self._add(OP_RY, q, p=th)
    def rz(self, q, th): self._add(OP_RZ, q, p=th)
    def cx(self, q1, q2): self._add(OP_CX, q1, q2)
    def cz(self, q1, q2): self._add(OP_CZ, q1, q2)
    def swap(self, q1, q2): self._add(OP_SWAP, q1, q2)
    def measure(self, q):
        self._add(OP_MEASURE, q)
        self.flush()
        return int(self.measure_out[q])
    def reset(self, q): self._add(OP_RESET, q)
    def reset_all(self):
        self.q_ptr=0; self.p_ptr=0
        self.gammas.fill(0); self.lambdas.fill(0)
        for i in range(self.num_qubits): self.gammas[i,0,0,0]=1.0
        for i in range(self.num_qubits+1): self.lambdas[i,0]=1.0
        self.measure_out.fill(0)
    def memory_usage(self): return self.gammas.nbytes + self.lambdas.nbytes
