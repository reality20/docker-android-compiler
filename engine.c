#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <complex.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#define MAX_BOND 4
#define MAX_SVD_DIM 8
#define TOL 1e-12
#define CHUNK_SIZE 32768

typedef double complex dcomplex;

static inline double norm_sq(dcomplex z) {
    return creal(z)*creal(z) + cimag(z)*cimag(z);
}

// Optimized Jacobi SVD
static void svd_jacobi_8x8(dcomplex* restrict A, dcomplex* restrict V, int m, int n) {
    memset(V, 0, n * n * sizeof(dcomplex));
    for (int i = 0; i < n; i++) V[i*n + i] = 1.0;

    int max_sweeps = 4;
    double eps = 1e-12;

    for (int sweep = 0; sweep < max_sweeps; sweep++) {
        int changed = 0;

        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 1; j < n; j++) {
                dcomplex alpha = 0;
                dcomplex beta = 0;
                dcomplex gamma = 0;

                for (int k = 0; k < m; k++) {
                    dcomplex a_ki = A[k*n + i];
                    dcomplex a_kj = A[k*n + j];
                    alpha += conj(a_ki) * a_ki;
                    beta  += conj(a_kj) * a_kj;
                    gamma += conj(a_ki) * a_kj;
                }

                double gamma_mag = cabs(gamma);
                if (gamma_mag < eps) continue;
                changed = 1;

                double r_alpha = creal(alpha);
                double r_beta = creal(beta);

                double diff = r_beta - r_alpha;
                double tau = diff / (2.0 * gamma_mag);
                double t = copysign(1.0 / (fabs(tau) + sqrt(1.0 + tau*tau)), tau);
                double c = 1.0 / sqrt(1.0 + t*t);
                dcomplex s = c * t * (gamma / gamma_mag);

                for (int k = 0; k < m; k++) {
                    dcomplex v_ki = A[k*n + i];
                    dcomplex v_kj = A[k*n + j];
                    A[k*n + i] = c * v_ki - conj(s) * v_kj;
                    A[k*n + j] = s * v_ki + c * v_kj;
                }

                for (int k = 0; k < n; k++) {
                    dcomplex v_ki = V[k*n + i];
                    dcomplex v_kj = V[k*n + j];
                    V[k*n + i] = c * v_ki - conj(s) * v_kj;
                    V[k*n + j] = s * v_ki + c * v_kj;
                }
            }
        }
        if (!changed) break;
    }
}

static void apply_1q(dcomplex* restrict tensors, int* restrict ranks, int i, dcomplex* restrict gate) {
    int L = ranks[i];
    int R = ranks[i+1];
    npy_intp site_stride = MAX_BOND * MAX_BOND * 2;
    dcomplex* node = tensors + i * site_stride;

    dcomplex g00 = gate[0], g01 = gate[1], g10 = gate[2], g11 = gate[3];

    // DEBUG
    // printf("apply_1q: i=%d, L=%d, R=%d. g00=%.2f+%.2fi\n", i, L, R, creal(g00), cimag(g00));

    for (int l = 0; l < L; l++) {
        int base = l * MAX_BOND * 2;
        for (int r = 0; r < R; r++) {
            int idx = base + r * 2;
            dcomplex v0 = node[idx];
            dcomplex v1 = node[idx+1];

            // printf("  Before: l=%d, r=%d. v0=%.2f, v1=%.2f\n", l, r, creal(v0), creal(v1));

            node[idx]   = g00*v0 + g01*v1;
            node[idx+1] = g10*v0 + g11*v1;

            // printf("  After: v0=%.2f, v1=%.2f\n", creal(node[idx]), creal(node[idx+1]));
        }
    }
}

static void apply_2q(dcomplex* restrict tensors, int* restrict ranks, int i, dcomplex* restrict gate) {
    int L = ranks[i];
    int M = ranks[i+1];
    int R = ranks[i+2];

    npy_intp stride = MAX_BOND * MAX_BOND * 2;
    dcomplex* n1 = tensors + i * stride;
    dcomplex* n2 = tensors + (i+1) * stride;

    dcomplex T12[64];
    memset(T12, 0, 64 * sizeof(dcomplex));

    for (int l = 0; l < L; l++) {
        for (int m = 0; m < M; m++) {
            int off1 = l * MAX_BOND * 2 + m * 2;
            dcomplex v1_0 = n1[off1];
            dcomplex v1_1 = n1[off1 + 1];

            for (int r = 0; r < R; r++) {
                 int off2 = m * MAX_BOND * 2 + r * 2;
                 dcomplex v2_0 = n2[off2];
                 dcomplex v2_1 = n2[off2 + 1];

                 T12[(l*4 + 0)*R + r] += v1_0 * v2_0;
                 T12[(l*4 + 1)*R + r] += v1_0 * v2_1;
                 T12[(l*4 + 2)*R + r] += v1_1 * v2_0;
                 T12[(l*4 + 3)*R + r] += v1_1 * v2_1;
            }
        }
    }

    dcomplex theta[64];
    for (int l = 0; l < L; l++) {
        for (int r = 0; r < R; r++) {
             int base = l*4*R + r;
             for (int p = 0; p < 4; p++) {
                 dcomplex val = 0;
                 val += gate[p*4 + 0] * T12[base + 0*R];
                 val += gate[p*4 + 1] * T12[base + 1*R];
                 val += gate[p*4 + 2] * T12[base + 2*R];
                 val += gate[p*4 + 3] * T12[base + 3*R];
                 theta[base + p*R] = val;
             }
        }
    }

    int n_rows = L * 2;
    int n_cols = 2 * R;
    dcomplex A[64];
    dcomplex V[64];

    for (int row=0; row<n_rows; row++) {
        int l = row >> 1;
        int p1 = row & 1;
        int row_off = (l*2 + p1)*2;
        for (int col=0; col<n_cols; col++) {
             int p2 = col / R;
             int r = col % R;
             A[row*n_cols + col] = theta[(row_off + p2)*R + r];
        }
    }

    svd_jacobi_8x8(A, V, n_rows, n_cols);

    double svals[MAX_SVD_DIM];
    for (int j = 0; j < n_cols; j++) {
        double sum = 0;
        for (int k = 0; k < n_rows; k++) sum += norm_sq(A[k*n_cols + j]);
        svals[j] = sqrt(sum);
    }

    int perm[MAX_SVD_DIM];
    for(int k=0; k<n_cols; k++) perm[k] = k;
    for (int k=0; k<n_cols-1; k++) {
        int max_idx = k;
        for (int j = k + 1; j < n_cols; j++) {
            if (svals[perm[j]] > svals[perm[max_idx]]) max_idx = j;
        }
        int tmp = perm[k]; perm[k] = perm[max_idx]; perm[max_idx] = tmp;
    }

    int new_bond = 0;
    for (int k = 0; k < n_cols; k++) {
        if (svals[perm[k]] > 1e-10) new_bond++;
    }
    if (new_bond > MAX_BOND) new_bond = MAX_BOND;
    ranks[i+1] = new_bond;

    memset(n1, 0, stride * sizeof(dcomplex));
    memset(n2, 0, stride * sizeof(dcomplex));

    for (int nb = 0; nb < new_bond; nb++) {
        int orig_col = perm[nb];
        double sv = svals[orig_col];
        double inv_sv = (sv > 1e-12) ? 1.0/sv : 0.0;

        for (int row = 0; row < n_rows; row++) {
             int l = row >> 1;
             int p1 = row & 1;
             n1[l*MAX_BOND*2 + nb*2 + p1] = A[row*n_cols + orig_col] * inv_sv;
        }

        for (int k = 0; k < n_cols; k++) {
             int p2 = k / R;
             int r = k % R;
             n2[nb*MAX_BOND*2 + r*2 + p2] = sv * conj(V[k*n_cols + orig_col]);
        }
    }
}

static PyObject* c_run_circuit(PyObject* self, PyObject* args) {
    PyObject *tensors_obj, *ranks_obj, *gate_cmds_obj, *gate_mats_obj;
    if (!PyArg_ParseTuple(args, "OOOO", &tensors_obj, &ranks_obj, &gate_cmds_obj, &gate_mats_obj)) return NULL;

    PyArrayObject *tensors_arr = (PyArrayObject*)PyArray_FROM_OTF(tensors_obj, NPY_CDOUBLE, NPY_ARRAY_INOUT_ARRAY);
    PyArrayObject *ranks_arr = (PyArrayObject*)PyArray_FROM_OTF(ranks_obj, NPY_INT32, NPY_ARRAY_INOUT_ARRAY);
    PyArrayObject *cmds_arr = (PyArrayObject*)PyArray_FROM_OTF(gate_cmds_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *mats_arr = (PyArrayObject*)PyArray_FROM_OTF(gate_mats_obj, NPY_CDOUBLE, NPY_ARRAY_IN_ARRAY);

    if (!tensors_arr || !ranks_arr || !cmds_arr || !mats_arr) return NULL;

    dcomplex* tensors = (dcomplex*)PyArray_DATA(tensors_arr);
    int* ranks = (int*)PyArray_DATA(ranks_arr);
    int* cmds = (int*)PyArray_DATA(cmds_arr);
    dcomplex* mats = (dcomplex*)PyArray_DATA(mats_arr);

    int total_gates = PyArray_DIM(cmds_arr, 0);
    int num_qubits = PyArray_DIM(tensors_arr, 0);

    // printf("run_circuit: total_gates=%d, num_qubits=%d\n", total_gates, num_qubits);

    int* batch_assign = (int*)malloc(CHUNK_SIZE * sizeof(int));
    int* sorted = (int*)malloc(CHUNK_SIZE * sizeof(int));
    int* counts = (int*)malloc((CHUNK_SIZE + 1) * sizeof(int));
    int* offsets = (int*)malloc((CHUNK_SIZE + 2) * sizeof(int));
    int* qubit_last = (int*)calloc(num_qubits, sizeof(int));

    for (int base = 0; base < total_gates; base += CHUNK_SIZE) {
        int end = base + CHUNK_SIZE;
        if (end > total_gates) end = total_gates;
        int size = end - base;

        memset(qubit_last, 0, num_qubits * sizeof(int));

        int max_batch = -1;

        for (int k = 0; k < size; k++) {
            int idx = base + k;
            int type = cmds[idx*4 + 0];
            int i = cmds[idx*4 + 1];
            int j = cmds[idx*4 + 2];

            int b = qubit_last[i];
            if (type == 2 && qubit_last[j] > b) b = qubit_last[j];

            batch_assign[k] = b;
            if (b > max_batch) max_batch = b;

            qubit_last[i] = b+1;
            if (type == 2) qubit_last[j] = b+1;
        }

        memset(counts, 0, (max_batch + 1) * sizeof(int));
        for (int k = 0; k < size; k++) counts[batch_assign[k]]++;

        int off = 0;
        for (int b = 0; b <= max_batch; b++) {
            offsets[b] = off;
            off += counts[b];
        }
        offsets[max_batch+1] = off;

        memset(counts, 0, (max_batch + 1) * sizeof(int));

        for (int k = 0; k < size; k++) {
            int b = batch_assign[k];
            sorted[offsets[b] + counts[b]++] = k;
        }

        // printf("Batch count: %d\n", max_batch+1);

        for (int b = 0; b <= max_batch; b++) {
            int start = offsets[b];
            int count = offsets[b+1] - start;

            // printf("  Batch %d: start=%d, count=%d\n", b, start, count);

            #pragma omp parallel for
            for (int c = 0; c < count; c++) {
                int k = sorted[start + c];
                int idx = base + k;

                int type = cmds[idx*4 + 0];
                int i = cmds[idx*4 + 1];
                dcomplex* g = mats + cmds[idx*4 + 3];

                if (type == 1) apply_1q(tensors, ranks, i, g);
                else apply_2q(tensors, ranks, i, g);
            }
        }
    }

    free(batch_assign);
    free(sorted);
    free(counts);
    free(offsets);
    free(qubit_last);

    Py_DECREF(tensors_arr);
    Py_DECREF(ranks_arr);
    Py_DECREF(cmds_arr);
    Py_DECREF(mats_arr);

    Py_RETURN_NONE;
}

static PyMethodDef BackendMethods[] = {
    {"run_circuit", c_run_circuit, METH_VARARGS, "Run circuit batch"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef backendmodule = {
    PyModuleDef_HEAD_INIT,
    "quantum_backend",
    "C Optimized Quantum Backend",
    -1,
    BackendMethods
};

PyMODINIT_FUNC PyInit_quantum_backend(void) {
    import_array();
    return PyModule_Create(&backendmodule);
}
