import numpy as np
import scipy.linalg
from numba import jit

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

def contract_1q(tensor, gate):
    # tensor: (D_L, 2, D_R)
    # gate: (2, 2)
    # out: (D_L, 2, D_R)
    # tensordot axes=([1], [1]) contracts physical dim.
    # tensor(i, l, j) * gate(k, l) -> (i, j, k). Transpose to (i, k, j)
    temp = np.tensordot(tensor, gate, axes=([1], [1]))
    return np.transpose(temp, (0, 2, 1))

def contract_2q_svd(left_tensor, right_tensor, gate, max_bond_dim):
    # left_tensor: (D_L, 2, D_mid) - CENTER (or implicitly absorbed)
    # right_tensor: (D_mid, 2, D_R)
    # gate: (2, 2, 2, 2) (out_l, out_r, in_l, in_r)

    # Merge tensors
    theta = np.tensordot(left_tensor, right_tensor, axes=(2, 0)) # (D_L, 2, 2, D_R)

    # Apply gate
    # theta(n, i, j, m) * gate(k, l, i, j) -> (n, m, k, l)
    theta_prime = np.tensordot(theta, gate, axes=([1, 2], [2, 3]))

    # Transpose to (n, k, l, m) -> (d_l, out_l, out_r, d_r)
    theta_prime = np.transpose(theta_prime, (0, 2, 3, 1))

    d_l = theta_prime.shape[0]
    d_r = theta_prime.shape[3]

    # Reshape for SVD: (d_l * 2, 2 * d_r)
    theta_mat = theta_prime.reshape(d_l * 2, 2 * d_r)

    # SVD using scipy for speed (LAPACK)
    U, S, Vh = scipy.linalg.svd(theta_mat, full_matrices=False, lapack_driver='gesdd')

    # Truncate
    k = min(max_bond_dim, len(S))
    U = U[:, :k]
    S = S[:k]
    Vh = Vh[:k, :]

    # Normalize S
    norm = np.linalg.norm(S)
    if norm > 1e-15:
        S = S / norm

    # Push S to the right -> Right tensor becomes the new center
    Vh = np.diag(S) @ Vh

    # Reshape
    new_left = U.reshape(d_l, 2, k)
    new_right = Vh.reshape(k, 2, d_r)

    return new_left, new_right

class QPU:
    def __init__(self, num_qubits, bond_dim, state='zeros'):
        self.num_qubits = num_qubits
        self.bond_dim = bond_dim
        self.gate_count = 0
        self.tensors = []
        self.center = 0 # Initially canonical at 0

        # Initialize product state |00...0>
        # (1, 2, 1) tensors
        # |0> state is (1, 0)
        # Tensor structure: Left dim 1, Right dim 1.
        # This is strictly canonical everywhere since it's a product state.
        # We set center at 0 arbitrarily.
        if state == 'zeros':
            self.reset_all()

    def reset_all(self):
        self.tensors = []
        for i in range(self.num_qubits):
            t = np.zeros((1, 2, 1), dtype=np.complex128)
            t[0, 0, 0] = 1.0
            self.tensors.append(t)
        self.center = 0
        self.gate_count = 0

    def move_center(self, target):
        if self.center == target:
            return

        # Sweep Right
        if self.center < target:
            for i in range(self.center, target):
                # QR on tensors[i] to make it Left Orthogonal
                # tensors[i]: (dL, 2, dR)
                dL, d, dR = self.tensors[i].shape
                mat = self.tensors[i].reshape(dL * d, dR)

                # Q: (dL*d, K), R: (K, dR)
                Q, R = scipy.linalg.qr(mat, mode='economic')

                K = R.shape[0]
                self.tensors[i] = Q.reshape(dL, d, K)

                # Absorb R into next tensor
                # tensors[i+1]: (dR, 2, dNext) -> (K, 2, dNext)
                self.tensors[i+1] = np.tensordot(R, self.tensors[i+1], axes=(1, 0))

            self.center = target

        # Sweep Left
        elif self.center > target:
            for i in range(self.center, target, -1):
                # LQ on tensors[i] to make it Right Orthogonal
                # tensors[i]: (dL, 2, dR)
                dL, d, dR = self.tensors[i].shape
                mat = self.tensors[i].reshape(dL, d * dR)

                # LQ = QR(M.T).T
                # M.T: (d*dR, dL)
                Q_t, R_t = scipy.linalg.qr(mat.T, mode='economic')
                # Q_t: (d*dR, K), R_t: (K, dL)
                # M = R_t.T @ Q_t.T = L @ Q
                L = R_t.T # (dL, K)
                Q = Q_t.T # (K, d*dR)

                K = L.shape[1]
                self.tensors[i] = Q.reshape(K, d, dR)

                # Absorb L into prev tensor
                # tensors[i-1]: (dPrev, 2, dL) -> (dPrev, 2, K)
                self.tensors[i-1] = np.tensordot(self.tensors[i-1], L, axes=(2, 0))

            self.center = target

    def apply_1q(self, gate, index):
        # 1-qubit gates preserve orthogonality, so center doesn't strictly need to move.
        # Just apply locally.
        self.tensors[index] = contract_1q(self.tensors[index], gate)
        self.gate_count += 1

    def apply_2q(self, gate, index1, index2):
        if index2 != index1 + 1:
            raise ValueError("Only nearest neighbor gates supported currently")

        # To optimize SVD truncation, the center should be at the bond being cut.
        # Moving center to index1 puts the weights in index1.
        self.move_center(index1)

        # contract_2q_svd splits weights into S, and pushes S to Vh (right tensor).
        # So left becomes Left-Orth, Right becomes Center.
        left, right = contract_2q_svd(self.tensors[index1], self.tensors[index2], gate, self.bond_dim)

        self.tensors[index1] = left
        self.tensors[index2] = right

        # Center is now at index2
        self.center = index2
        self.gate_count += 1

    def measure(self, index):
        """
        Projective measurement in Z-basis.
        Returns 0 or 1.
        """
        # Move center to index so that tensor represents local state correctly normalized
        self.move_center(index)

        T = self.tensors[index] # (dL, 2, dR)

        # Prob(0) = sum(|T[:, 0, :]|^2)
        p0 = np.sum(np.abs(T[:, 0, :])**2)
        p1 = 1.0 - p0

        # Numerical stability check
        if p0 < 0: p0 = 0.0
        if p0 > 1: p0 = 1.0

        outcome = 0
        if np.random.random() > p0:
            outcome = 1

        # Collapse state
        # If 0, zero out the 1-component
        # If 1, zero out the 0-component
        if outcome == 0:
            T[:, 1, :] = 0.0
            norm = np.sqrt(p0)
        else:
            T[:, 0, :] = 0.0
            norm = np.sqrt(1.0 - p0) # Recalc from p0 to avoid slight drift

        if norm < 1e-15:
            # Should not happen if prob > 0
            norm = 1.0

        T /= norm
        self.tensors[index] = T
        # Center remains at index

        return outcome

    def reset(self, index):
        """
        Resets qubit to |0>.
        Effectively: Measure. If 1, apply X.
        """
        m = self.measure(index)
        if m == 1:
            self.x(index)

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
            self.h(c)
            self.h(t)
            self.apply_2q(CX, t, c)
            self.h(c)
            self.h(t)
        else:
             raise ValueError(f"CX not supported for non-adjacent qubits {c}, {t}")

    def cz(self, i, j):
        if j == i + 1:
            self.apply_2q(CZ, i, j)
        elif i == j + 1:
            self.apply_2q(CZ, j, i)
        else:
             raise ValueError(f"CZ not supported for non-adjacent qubits {i}, {j}")

    def swap(self, i, j):
        if j == i + 1:
            self.apply_2q(SWAP, i, j)
        elif i == j + 1:
            self.apply_2q(SWAP, j, i)
        else:
             raise ValueError(f"SWAP not supported for non-adjacent qubits {i}, {j}")

    def memory_usage(self):
        mem = 0
        for t in self.tensors:
            mem += t.nbytes
        return mem
