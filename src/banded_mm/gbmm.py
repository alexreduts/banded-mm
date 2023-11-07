"""
General banded matrix multiplication

"""

import numpy as np

from banded_mm.matrix_utils import banded_matrix_generator, binary_grid
from banded_mm.naive_mm import naive_banded_mm

# General blocked banded matrix multiplication (Only A is banded)
def _gbmm_BLK_outer(
        C: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        ku: int,
        kl: int
    ) -> np.ndarray:

    #Partition
    CL_ = slice(0, 0)
    CR_ = slice(CL_.stop, C.shape[1])

    BL_ = slice(0, 0)
    BR_ = slice(BL_.stop, B.shape[1])
    
    while CL_.stop < C.shape[1]:
        # Tune for HW
        c = 2

        # Repartition
        C0_ = slice(CL_.start, CL_.stop)
        C1_ = slice(C0_.stop, C0_.stop+c)
        C2_ = slice(C1_.stop, CR_.stop)

        B0_ = slice(BL_.start, BL_.stop)
        B1_ = slice(B0_.stop, B0_.stop+c)
        B2_ = slice(B1_.stop, BR_.stop)

        # inner loop
        C[:, C1_] = _gbmm_BLK_inner(C[:, C1_], A, B[:, B1_], ku, kl)

        # Adjust partition
        CL_ = slice(C0_.start, C1_.stop)
        CR_ = slice(CL_.stop, C2_.stop)

        BL_ = slice(B0_.start, B1_.stop)
        BR_ = slice(BL_.stop, B2_.stop)

    return C


def _gbmm_BLK_inner(
        E: np.ndarray,
        A: np.ndarray,
        D: np.ndarray,
        ku: int,
        kl: int
    ) -> np.ndarray:

    #Partition
    ET_ = slice(0, 0)
    EM_ = slice(ET_.stop, ET_.stop+kl)
    EB_ = slice(EM_.stop, E.shape[0])

    DT_ = slice(0, 0)
    DB_ = slice(DT_.stop, D.shape[0])

    AT_ = slice(0, 0)
    AM_ = slice(AT_.stop, AT_.stop+kl)
    AB_ = slice(AM_.stop, A.shape[0])

    AL_ = slice(0, 0)
    AR_ = slice(AL_.stop, A.shape[1])

    while E[ET_, :].shape[0] < E.shape[0]:
        
        # Tune for Hardware
        b = 4

        # Repartition
        D0_ = slice(DT_.start, DT_.stop)
        D1_ = slice(D0_.stop, D0_.stop+b)
        D2_ = slice(D1_.stop, DB_.stop)

        Ax0_ = slice(AL_.start, AL_.stop)
        Ax1_ = slice(Ax0_.stop, Ax0_.stop+b)
        Ax2_ = slice(Ax1_.stop, AR_.stop)

        E0_ = slice(ET_.start, ET_.stop)

        if D[D0_, :].shape[0] < (ku+1):
            #E1_ has 0 rows
            E1_ = slice(EM_.start, EM_.start)
            #A11 is emtpy
            A1x_ = slice(AM_.start, AM_.start)
        else:
            #E1_ has b rows
            E1_ = slice(EM_.start, EM_.start+b)
            #A11 is bxb
            A1x_ = slice(AM_.start, AM_.start+b)

            E[E1_, :] = E[E1_, :] + A[A1x_, Ax1_] @ D[D1_, :]

        if D[D0_, :].shape[0] > (A.shape[1]-kl-1):
            #E3_ has 0 rows
            E3_ = slice(EB_.start+b, EB_.start+b)
            A3x_ = slice(AB_.start+b, AB_.start+b)
        else:
            #E3 has b rows
            E3_ = slice(EB_.start, EB_.start+b)
            #A33 is bxb
            A3x_ = slice(AB_.start, AB_.start+b)

            E[E3_, :] = E[E3_, :] + A[A3x_, Ax1_] @ D[D1_, :]

        E2_ = slice(E1_.stop, E3_.start)
        A2x_ = slice(A1x_.stop, A3x_.start)

        E4_ = slice(E3_.stop, EB_.stop)
        A4x_ = slice(A3x_.stop, AB_.stop)

        E[E2_, :] = E[E2_, :] + A[A2x_, Ax1_] @ D[D1_, :]

        

        # Adjust partition
        ET_ = slice(E0_.start, E1_.stop)
        EM_ = slice(E2_.start, E3_.stop)
        EB_ = slice(E4_.start, E4_.stop)

        AT_ = slice(0, A1x_.stop)
        AM_ = slice(A2x_.start, A3x_.stop)
        AB_ = slice(A4x_.start, A4x_.stop)
        
        AL_ = slice(0, Ax1_.stop)
        AR_ = slice(Ax2_.start, Ax2_.stop)

        DT_ = slice(0, D1_.stop)
        DB_ = slice(DT_.stop, D2_.stop)

    return E


    
def gbmm_BLK(
        C: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        ku: int,
        kl: int
    ):
    C = _gbmm_BLK_outer(C, A, B, ku, kl)
    return C

#A = banded_matrix_generator(8, 2, 2)
#B = banded_matrix_generator(8, 2, 2)
#C = np.zeros((A.shape[0], B.shape[1]))
#H = gbmm_BLK(C, A, B, 2, 2)
#T = np.matmul(A,B)

#print("H", H)
#print("T", T)
#print("Diff\n", H-T)
#binary_grid(H-T)
#assert np.allclose(H, T)

# General banded times banded matrix multiplication (A & B banded)
def _gbmm_BB_outer(
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

    #Partition
    CL_ = slice(0, 0)
    CR_ = slice(CL_.stop, C.shape[1])

    BL_ = slice(0, 0)
    BR_ = slice(BL_.stop, B.shape[1])
    
    while CL_.stop < C.shape[1]:
        # Tune for HW
        c = 5

        # Repartition
        C0_ = slice(CL_.start, CL_.stop)
        C1_ = slice(C0_.stop, C0_.stop+c)
        C2_ = slice(C1_.stop, CR_.stop)

        B0_ = slice(BL_.start, BL_.stop)
        B1_ = slice(B0_.stop, B0_.stop+c)
        B2_ = slice(B1_.stop, BR_.stop)

        C_1 = slice(max(0, C1_.start - ku_C), min(C.shape[1], C1_.stop + kl_C))
        B_1 = slice(max(0, B1_.start - ku_B), min(B.shape[1], B1_.stop + kl_B))

        ku = ku_A + (C_1.start - B_1.start)
        kl = kl_A - (C_1.start - B_1.start)

        # inner loop
        C[C_1, C1_] = _gbmm_BB_inner(C[C_1, C1_], A[C_1, B_1], B[B_1, B1_], ku, kl)

        binary_grid(C[C_1, C1_])

        # Adjust partition
        CL_ = slice(C0_.start, C1_.stop)
        CR_ = slice(CL_.stop, C2_.stop)

        BL_ = slice(B0_.start, B1_.stop)
        BR_ = slice(BL_.stop, B2_.stop)

    return C


def _gbmm_BB_inner(
        E: np.ndarray,
        A: np.ndarray,
        D: np.ndarray,
        ku: int,
        kl: int
    ) -> np.ndarray:

    #Partition
    ET_ = slice(0, 0)
    EM_ = slice(ET_.stop, ET_.stop+kl)
    EB_ = slice(EM_.stop, E.shape[0])

    DT_ = slice(0, 0)
    DB_ = slice(DT_.stop, D.shape[0])

    AT_ = slice(0, 0)
    AM_ = slice(AT_.stop, AT_.stop+kl)
    AB_ = slice(AM_.stop, A.shape[0])

    AL_ = slice(0, 0)
    AR_ = slice(AL_.stop, A.shape[1])

    while E[ET_, :].shape[0] < E.shape[0]:
        
        # Tune for Hardware
        b = 4

        # Repartition
        D0_ = slice(DT_.start, DT_.stop)
        D1_ = slice(D0_.stop, D0_.stop+b)
        D2_ = slice(D1_.stop, DB_.stop)

        Ax0_ = slice(AL_.start, AL_.stop)
        Ax1_ = slice(Ax0_.stop, Ax0_.stop+b)
        Ax2_ = slice(Ax1_.stop, AR_.stop)

        E0_ = slice(ET_.start, ET_.stop)

        dim_D0 = D[D0_, :].shape[0]
        if D[D0_, :].shape[0] < (ku+1):
            #E1_ has 0 rows
            E1_ = slice(EM_.start, EM_.start)
            #A11 is emtpy
            A1x_ = slice(AM_.start, AM_.start)
        else:
            #E1_ has b rows
            E1_ = slice(EM_.start, EM_.start+b)
            #A11 is bxb
            A1x_ = slice(AM_.start, AM_.start+b)

            E[E1_, :] = E[E1_, :] + A[A1x_, Ax1_] @ D[D1_, :]

        if D[D0_, :].shape[0] > (A.shape[1]-kl-1):
            #E3_ has 0 rows
            E3_ = slice(EB_.start+b, EB_.start+b)
            #A33 is empty
            A3x_ = slice(AB_.start+b, AB_.start+b)
        else:
            #E3 has b rows
            E3_ = slice(EB_.start, EB_.start+b)
            #A33 is bxb
            A3x_ = slice(AB_.start, AB_.start+b)

            E[E3_, :] = E[E3_, :] + A[A3x_, Ax1_] @ D[D1_, :]

        E2_ = slice(E1_.stop, E3_.start)
        A2x_ = slice(A1x_.stop, A3x_.start)

        E4_ = slice(E3_.stop, EB_.stop)
        A4x_ = slice(A3x_.stop, AB_.stop)

        E[E2_, :] = E[E2_, :] + A[A2x_, Ax1_] @ D[D1_, :]

        # Adjust partition
        ET_ = slice(E0_.start, E1_.stop)
        EM_ = slice(E2_.start, E3_.stop)
        EB_ = slice(E4_.start, E4_.stop)

        AT_ = slice(0, A1x_.stop)
        AM_ = slice(A2x_.start, A3x_.stop)
        AB_ = slice(A4x_.start, A4x_.stop)
        
        AL_ = slice(0, Ax1_.stop)
        AR_ = slice(Ax2_.start, Ax2_.stop)

        DT_ = slice(0, D1_.stop)
        DB_ = slice(DT_.stop, D2_.stop)

    return E

def  gbmm_BB(
        C: np.ndarray,
        A: np.ndarray,
        ku_A: int,
        kl_A: int,
        B: np.ndarray,
        ku_B: int,
        kl_B: int
    ):
    C = _gbmm_BB_outer(C, A, ku_A, kl_A, B, ku_B, kl_B)
    return C

A = banded_matrix_generator(50, 2, 7)
B = banded_matrix_generator(50, 1, 6)
C = np.zeros((A.shape[0], B.shape[1]))
H = gbmm_BB(C, A, 2, 7, B, 1, 6)
#H = gbmm_BLK(C, A, B, 2, 7)
#binary_grid(H)
T = np.matmul(A,B)

#print("H", H)
#print("T", T)
#print("Diff\n", H-T)
binary_grid(H-T)
assert np.allclose(H, T)