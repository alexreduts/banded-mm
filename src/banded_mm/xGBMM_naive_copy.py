"""
Naive copy implementation of xGBMM
"""

import numpy as np
import cupy as cp

from banded_mm.matrix_utils import banded_matrix_generator

# General banded times banded matrix multiplication (A & B banded)
def _xGBMM_outer(
        C: np.ndarray,
        A: np.ndarray,
        ku_A: int,
        kl_A: int,
        B: np.ndarray,
        ku_B: int,
        kl_B: int,
        block_size_outer: int,
        block_size_inner: int
    ) -> np.ndarray:

    ku_C = ku_A + ku_B
    kl_C = kl_A + kl_B

    n = C.shape[1]

    # Initialize Partition
    Cx1 = slice(0, block_size_outer)
    Bx1 = slice(0, block_size_outer)

    while Cx1.start < n:

        # Shrink blocks to bandwidth
        C1x = slice(max(0, Cx1.start - ku_C), min(n, Cx1.stop + kl_C))
        B1x = slice(max(0, Bx1.start - ku_B), min(n, Bx1.stop + kl_B))

        # Adjust number of upper and lower bands matching subblocks
        ku = ku_A + (C1x.start - B1x.start)
        kl = kl_A - (C1x.start - B1x.start)

        # inner loop
        C[C1x, Cx1] = _xGBMM_inner(
            C[C1x, Cx1],
            A[C1x, B1x],
            B[B1x, Bx1],
            ku, kl,
            block_size_inner,
        )

        # Adjust Partition
        Cx1 = slice(Cx1.start+block_size_outer, Cx1.stop+block_size_outer)
        Bx1 = slice(Bx1.start+block_size_outer, Bx1.stop+block_size_outer)

    return C

def _slicer(
        iter: int,
        k: int,
        m: int,
        ku: int,
        kl: int,
        block_size_inner: int
):
    """
    Explicitly calculate the borders of the blocks to be multiplied
    based on the iteration specified
    """
    
    # Position
    pos = iter*block_size_inner

    # Band limits relative to position
    band_limit_upper = max(0, pos - ku)
    band_limit_lower = min(pos + kl + block_size_inner, m)
    
    D1 = slice(pos, min(k, pos+block_size_inner))

    if pos < (ku+1):
        A1 = slice(band_limit_upper, band_limit_upper)
    else:
        A1 = slice(band_limit_upper, band_limit_upper+block_size_inner)

    if pos > (k-kl-1):
        A3 = slice(band_limit_lower, band_limit_lower)
    else:
        A3 = slice(band_limit_lower-block_size_inner, band_limit_lower)

    A2 = slice(A1.stop, A3.start)

    return D1, A1, A2, A3

def _xGBMM_inner(
        E: cp.ndarray,
        A: cp.ndarray,
        D: cp.ndarray,
        ku: int,
        kl: int,
        block_size_inner,
    ) -> cp.ndarray:

    k = D.shape[0]
    n = D.shape[1]
    m = A.shape[0]

    # Number of blocks to compute
    num_blocks = -(-k//block_size_inner) #Rounded up

    # Iteration step i = 0
    D1_cur, A1_cur, A2_cur, A3_cur = _slicer(0, k, m, ku, kl, block_size_inner)
        
    if A1_cur.stop > A1_cur.start:
        E[A1_cur, :] = E[A1_cur, :] + A[A1_cur,D1_cur] @ D[D1_cur, :]

    if A2_cur.stop > A2_cur.start:
        E[A2_cur, :] = E[A2_cur, :] + A[A2_cur,D1_cur] @ D[D1_cur, :]

    if A3_cur.stop > A3_cur.start:
        E[A3_cur, :] = E[A3_cur, :] + A[A3_cur,D1_cur] @ D[D1_cur, :]

    # Iteration step i = 1 to i = num_blocks-1
    for i in range(1, num_blocks):

        D1_cur, A1_cur, A2_cur, A3_cur = _slicer(i, k, m, ku, kl, block_size_inner)
            
        if A1_cur.stop > A1_cur.start:
            E[A1_cur, :] = E[A1_cur, :] + A[A1_cur,D1_cur] @ D[D1_cur, :]

        if A2_cur.stop > A2_cur.start:
            E[A2_cur, :] = E[A2_cur, :] + A[A2_cur,D1_cur] @ D[D1_cur, :]

        if A3_cur.stop > A3_cur.start:
            E[A3_cur, :] = E[A3_cur, :] + A[A3_cur,D1_cur] @ D[D1_cur, :]

    return E

def  xGBMM_naive_copy(
        A: np.ndarray,
        ku_A: int,
        kl_A: int,
        B: np.ndarray,
        ku_B: int,
        kl_B: int,
        block_size_outer,
        block_size_inner
    ):
    C = np.zeros((A.shape[0], B.shape[1]))
    C = _xGBMM_outer(C, A, ku_A, kl_A, B, ku_B, kl_B, block_size_outer, block_size_inner)
    return C

if __name__ == "__main__":

    import sys
    
    flagged = False
    try:
        if sys.argv[1] == "--profiling":
            flagged = True
    except:
        pass

    if flagged:
        print("Profiling Setup Used")
        print("Generating band matrices")
        A = banded_matrix_generator(10000, 2400, 2900)
        B = banded_matrix_generator(10000, 3000, 800)

        print("Calculating gbmm_gpu")
        C = xGBMM_naive_copy(A, 2400, 2900, B, 3000, 800, 3000, 3000)
    else:
        print("Debug Setup Used")
        print("Generating band matrices")
        #A = banded_matrix_generator(10, 2, 2)
        #B = banded_matrix_generator(10, 0, 2)
        A = banded_matrix_generator(10, 1, 1)
        B = banded_matrix_generator(10, 1, 1)

        print("Calculating gbmm_gpu")
        #C = gbmm_gpu(A, 2, 2, B, 0, 2, 3, 2)
        C = xGBMM_naive_copy(A, 1, 1, B, 1, 1, 3, 2)

    print("Calculating Ref with numpy")
    T = A @ B
    print(T)
    print(C)
    assert np.allclose(C, T)
    print("Correct Result computed")