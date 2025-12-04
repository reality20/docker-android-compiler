import numpy as np
import numba
from numba import jit, njit, prange

# Define Gates
I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)

def rx(theta):
    return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                     [-1j*np.sin(theta/2), np.cos(theta/2)]], dtype=np.complex128)

def ry(theta):
    return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                     [np.sin(theta/2), np.cos(theta/2)]], dtype=np.complex128)

def rz(theta):
    return np.array([[np.exp(-1j*theta/2), 0],
                     [0, np.exp(1j*theta/2)]], dtype=np.complex128)

# 2-qubit gates
CX = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 0, 1],
               [0, 0, 1, 0]], dtype=np.complex128).reshape(2, 2, 2, 2)

CZ = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, -1]], dtype=np.complex128).reshape(2, 2, 2, 2)

SWAP = np.array([[1, 0, 0, 0],
                 [0, 0, 1, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1]], dtype=np.complex128).reshape(2, 2, 2, 2)

@jit(nopython=True, fastmath=True)
def contract_1q(tensor, gate):
    # tensor: (D_L, 2, D_R)
    # gate: (2, 2)
    # out: (D_L, 2, D_R)
    D_L = tensor.shape[0]
    D_R = tensor.shape[2]
    out = np.zeros_like(tensor)
    for i in range(D_L):
        for j in range(D_R):
            for k in range(2):
                for l in range(2):
                    out[i, k, j] += gate[k, l] * tensor[i, l, j]
    return out

def contract_2q_svd(left_tensor, right_tensor, gate, max_bond_dim):
    # left_tensor: (D_L, 2, D_mid)
    # right_tensor: (D_mid, 2, D_R)
    # gate: (2, 2, 2, 2) (input_left, input_right, output_left, output_right) - Standard is usually (out_L, out_R, in_L, in_R)
    # Let's align gate shape: gate[out_l, out_r, in_l, in_r]

    # Merge tensors
    # theta: (D_L, 2, 2, D_R)
    theta = np.tensordot(left_tensor, right_tensor, axes=(2, 0))

    # Apply gate
    # gate is 4x4 usually, let's reshape gate to (2, 2, 2, 2) where indices are (out_l, out_r, in_l, in_r)
    # theta indices: (d_l, in_l, in_r, d_r)
    # result: (d_l, out_l, out_r, d_r)

    # Using einsum for clarity, then optimize if needed.
    # n=d_l, i=in_l, j=in_r, m=d_r
    # k=out_l, l=out_r
    # gate[k, l, i, j] * theta[n, i, j, m] -> [n, k, l, m]
    theta_prime = np.tensordot(theta, gate, axes=([1, 2], [2, 3]))
    # tensordot output will be (d_l, d_r, out_l, out_r). We need (d_l, out_l, out_r, d_r)
    theta_prime = np.transpose(theta_prime, (0, 2, 3, 1))

    d_l = theta_prime.shape[0]
    d_r = theta_prime.shape[3]

    # Reshape for SVD: (d_l * 2, 2 * d_r)
    theta_mat = theta_prime.reshape(d_l * 2, 2 * d_r)

    # SVD
    U, S, Vh = np.linalg.svd(theta_mat, full_matrices=False)

    # Truncate
    k = min(max_bond_dim, len(S))
    # Keep only singular values > threshold? For now just fixed bond dim or full

    U = U[:, :k]
    S = S[:k]
    Vh = Vh[:k, :]

    # Absorb S into Vh (canonical form to the right usually, or split)
    # Let's split sqrt(S) for symmetry or put S to right.
    # Standard mixed canonical usually keeps orthogonality. Let's multiply S into Vh (Right Canonicalish update for next step? Or Left?)
    # For simple time evolution, usually we keep Vidal form (Gamma-Lambda-Gamma) or just normalize.
    # Here I'll just push S to the right tensor to maintain valid MPS form generally.

    Vh = np.diag(S) @ Vh

    # Reshape back
    # U: (d_l * 2, k) -> (d_l, 2, k)
    new_left = U.reshape(d_l, 2, k)
    # Vh: (k, 2 * d_r) -> (k, 2, d_r)
    new_right = Vh.reshape(k, 2, d_r)

    return new_left, new_right

class QPU:
    def __init__(self, num_qubits, bond_dim, state='zeros'):
        self.num_qubits = num_qubits
        self.bond_dim = bond_dim
        self.gate_count = 0
        self.tensors = []

        # Initialize product state |00...0>
        # (1, 2, 1) tensors
        if state == 'zeros':
            for i in range(num_qubits):
                t = np.zeros((1, 2, 1), dtype=np.complex128)
                t[0, 0, 0] = 1.0
                self.tensors.append(t)

    def apply_1q(self, gate, index):
        self.tensors[index] = contract_1q(self.tensors[index], gate)
        self.gate_count += 1

    def apply_2q(self, gate, index1, index2):
        if index2 != index1 + 1:
            raise ValueError("Only nearest neighbor gates supported currently")

        left, right = contract_2q_svd(self.tensors[index1], self.tensors[index2], gate, self.bond_dim)
        self.tensors[index1] = left
        self.tensors[index2] = right
        self.gate_count += 1

    # Gate wrappers
    def h(self, idx): self.apply_1q(H, idx)
    def s(self, idx): self.apply_1q(S, idx)
    def t(self, idx): self.apply_1q(T, idx)
    def x(self, idx): self.apply_1q(X, idx)
    def y(self, idx): self.apply_1q(Y, idx)
    def z(self, idx): self.apply_1q(Z, idx)
    def rx(self, idx, theta): self.apply_1q(rx(theta), idx)
    def ry(self, idx, theta): self.apply_1q(ry(theta), idx)
    def rz(self, idx, theta): self.apply_1q(rz(theta), idx)

    def cx(self, c, t):
        if c == t - 1:
            self.apply_2q(CX, c, t)
        elif t == c - 1:
            # Swap control/target logic or use SWAP?
            # CX(j, i) = (H_j H_i) CX(i, j) (H_j H_i)
            self.h(c)
            self.h(t)
            self.apply_2q(CX, t, c)
            self.h(c)
            self.h(t)
        else:
             # Implement SWAP chain if needed, but for now stick to NN
             pass

    def cz(self, i, j):
        if j == i + 1:
            self.apply_2q(CZ, i, j)
        elif i == j + 1:
            self.apply_2q(CZ, j, i)

    def swap(self, i, j):
        if j == i + 1:
            self.apply_2q(SWAP, i, j)
        elif i == j + 1:
            self.apply_2q(SWAP, j, i)

    def memory_usage(self):
        # Estimate bytes
        mem = 0
        for t in self.tensors:
            mem += t.nbytes
        return mem
