"""
Experimental file to try out new things

"""
from dataclasses import dataclass

from math import ceil
import numpy as np
import matplotlib.pyplot as plt

from matrix_utils import banded_matrix_generator
from naive_mm import naive_banded_mm




# A = banded_matrix_generator(10, 3, 3)
# B = banded_matrix_generator(10, 2, 2)
# H = naive_blocked_banded_mm(A, 3, 3, B, 2, 2, 5)
# T = np.matmul(A,B)

# This doesn't work
A = banded_matrix_generator(8, 2, 2)
B = banded_matrix_generator(8, 2, 2)
H = naive_blocked_banded_mm(A, 5, B, 5, 2)
T = np.matmul(A,B)

print("Diff\n", H-T)
binary_grid(H-T)
assert np.allclose(H, T)

H2 = naive_blocked_mm(A, B, 5)
assert np.allclose(H2, T)





#####################################################
# COO - Coordiante List
# (i, j, val)
#####################################################

#class COO:
#    rowidx: np.ndarray([nnz], dtype=np.int32)
#    colidx: np.ndarray([nnz], dtype=np.int32)
#    data: np.ndarray([nnz], dtype=np.float64)
#    mat: np.ndarray([nnz,3])

#####################################################
## CSR = Compressed Sparse Row (CSC = Compressed Sparse Column)
#A.CSR MxM x:vector N 
#A.rowptr[M+1], A.colindices[nnz], A.data[nnz]
#y = np.empty((M,))

#for i in range(M):
#    y[i] = 0
#    for j in range(A.rowptr[i], A.rowptrs[i+1]):
#        y[i] += A.data[j] * x[A.indicies[j]]


#class CSR:
#    value: np.ndarray
#    rowptr: np.ndarray
#    colidx: np.ndarray

#    def __init__(self, nnz, m) -> None:
#        self.rowptr = np.empty(1)
#        self.colidx = np.empty(0)
#        self.value = np.empty(0)

#def store_banded_as_CSR(matrix: np.ndarray, bandwidth: int) -> CSR:
#    nnz = matrix.shape[0]
#    for i in range(1, int((bandwidth-1)/2)+1):
#        nnz += 2*(matrix.shape[0]-i)
#    print("NNZ: ", nnz)

#    compressed = CSR(nnz, matrix.shape[1])
#    compressed.rowptr[0] = 0
#    k = 0
#    for i in range(0,matrix.shape[0]):
#        for j in range(0,matrix.shape[0]):
#            if matrix[i][j] != 0:
#                compressed.value = np.append(compressed.value, matrix[i][j])
#                compressed.colidx = np.append(compressed.colidx, j)
#                k += 1
#        compressed.rowptr = np.append(compressed.rowptr, k)
#
#    return compressed

#D_csr = store_banded_as_CSR(D, 3)
#print("CSR\n", D_csr.rowptr, "\n", D_csr.colidx, "\n", D_csr.value)

#Alt = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]])
#Alt_csr = store_banded_as_CSR(Alt, 1)
#print("Alt\n", Alt)
#print("Alt_csr\n", Alt_csr.rowptr, "\n", Alt_csr.colidx, "\n", Alt_csr.value)

#def CSRmul(A: CSR, B: CSR) -> CSR:
    # nnz
#    pass        


