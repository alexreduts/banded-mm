import numpy as np
import cupy as cp
import cupyx as cpx

from banded_mm.matrix_utils import banded_matrix_generator

# General banded times banded matrix multiplication (A & B banded)
def _gbmm_gpu_outer(
        C: np.ndarray,
        A: np.ndarray,
        ku_A: int,
        kl_A: int,
        B: np.ndarray,
        ku_B: int,
        kl_B: int
    ) -> np.ndarray:

    ku_C = ku_A + ku_B
    kl_C = kl_A + kl_B

    n = C.shape[1]

    # Initialize Iterators
    C_iter = 0
    B_iter = 0

    while C_iter < n:
        # Tune for HW
        c = 4

        # Partition
        C_col = slice(C_iter, C_iter+c)
        B_col = slice(B_iter, B_iter+c)

        # Shrink blocks to bandwidth
        C_row = slice(max(0, C_col.start - ku_C), min(n, C_col.stop + kl_C))
        B_row = slice(max(0, B_col.start - ku_B), min(n, B_col.stop + kl_B))

        # Adjust number of upper and lower bands matching subblocks
        ku = ku_A + (C_row.start - B_row.start)
        kl = kl_A - (C_row.start - B_row.start)

        # inner loop
        C[C_row, C_col] = _gbmm_gpu_inner(C[C_row, C_col], A[C_row, B_row], B[B_row, B_col], ku, kl)

        # Adjust Iterators
        C_iter += c
        B_iter += c

    return C


def _gbmm_gpu_inner(
        E: np.ndarray,
        A: np.ndarray,
        D: np.ndarray,
        ku: int,
        kl: int
    ) -> np.ndarray:

    m = E.shape[0]
    k = D.shape[0]

    # Initialize Iterators
    E_iter = 0
    E_block = kl
    D_iter = 0

    while E_iter < m:
        b = 3

        # Partition
        D1_ = slice(D_iter, D_iter+b)

        if D_iter < (ku+1):
            E1_ = slice(E_iter, E_iter)

        else:
            E1_ = slice(E_iter, E_iter+b)
            E[E1_, :] = cp.asnumpy(cp.asarray(E[E1_, :]) + cp.matmul(cp.asarray(A[E1_, D1_]), cp.asarray(D[D1_, :])))
            E_iter += b

        if D_iter > (k-kl-1):
            E3_ = slice(E_block+b, E_block+b)

        else:
            E3_ = slice(E_block, E_block+b)
            E[E3_, :] = cp.asnumpy(cp.asarray(E[E3_, :]) + cp.matmul(cp.asarray(A[E3_, D1_]),cp.asarray(D[D1_, :])))

        E2_ = slice(E1_.stop, E3_.start)
        E[E2_, :] = cp.asnumpy(cp.asarray(E[E2_, :]) + cp.matmul(cp.asarray(A[E2_, D1_]),cp.asarray(D[D1_, :])))

        # Adjust partition
        E_block += b
        D_iter += b
        assert E_block == (D_iter+kl)

    return E

def  gbmm_gpu(
        A: np.ndarray,
        ku_A: int,
        kl_A: int,
        B: np.ndarray,
        ku_B: int,
        kl_B: int
    ):
    C = np.zeros((A.shape[0], B.shape[1]))
    C = _gbmm_gpu_outer(C, A, ku_A, kl_A, B, ku_B, kl_B)
    return C

if __name__ == "__main__":

    A = banded_matrix_generator(24, 2, 1)
    B = banded_matrix_generator(24, 3, 7)
    C = gbmm_gpu(A, 2, 1, B, 3, 7)

    T = A @ B
    #print(C-T)
    assert np.allclose(C, T)
    print("Correct Result computed")