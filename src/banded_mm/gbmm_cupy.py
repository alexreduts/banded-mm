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

    # Tune for HW
    c = 4

    # Initialize Partition
    Cx1 = slice(0, c)
    Bx1 = slice(0, c)

    while Cx1.start < n:

        # Shrink blocks to bandwidth
        C1x = slice(max(0, Cx1.start - ku_C), min(n, Cx1.stop + kl_C))
        B1x = slice(max(0, Bx1.start - ku_B), min(n, Bx1.stop + kl_B))

        # Adjust number of upper and lower bands matching subblocks
        ku = ku_A + (C1x.start - B1x.start)
        kl = kl_A - (C1x.start - B1x.start)

        # inner loop
        C[C1x, Cx1] = _gbmm_gpu_inner(C[C1x, Cx1], A[C1x, B1x], B[B1x, Bx1], ku, kl)

        # Adjust Partition
        Cx1 = slice(Cx1.start+c, Cx1.stop+c)
        Bx1 = slice(Bx1.start+c, Bx1.stop+c)

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
    #E_iter = 0
    #E_block = kl
    #D_iter = 0

    # Tune for HW
    b = 3

    # Inititalize Partition
    D1x = slice(0, b)

    E1x = slice(0, 0)
    E2x = slice(0, kl)
    E3x = slice(kl, kl)

    while E1x.start < m:
    
        if D1x.start < (ku+1):
            # Repartition
            E1x = slice(E1x.start, E1x.start)

        else:
            # Repartition
            E1x = slice(E1x.start, E1x.start+b)
            # Compute Kernel on GPU
            E[E1x, :] = cp.asnumpy(cp.asarray(E[E1x, :]) + cp.matmul(cp.asarray(A[E1x, D1x]), cp.asarray(D[D1x, :])))
            # Adjust partition
            E1x = slice(E1x.start+b, E1x.stop)

        if D1x.start > (k-kl-1):
            # Repartition
            E3x = slice(E3x.start+b, E3x.start+b)

        else:
            # Repartition
            E3x = slice(E3x.start, E3x.start+b)
            # Compute Kernel on GPU
            E[E3x, :] = cp.asnumpy(cp.asarray(E[E3x, :]) + cp.matmul(cp.asarray(A[E3x, D1x]),cp.asarray(D[D1x, :])))

        # Repartition
        E2x = slice(E1x.stop, E3x.start)
        # Compute Kernel on GPU
        E[E2x, :] = cp.asnumpy(cp.asarray(E[E2x, :]) + cp.matmul(cp.asarray(A[E2x, D1x]),cp.asarray(D[D1x, :])))

        # Adjust partition
        D1x = slice(D1x.start+b, D1x.stop+b)
        E3x = slice(E3x.start+b, E3x.stop)



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