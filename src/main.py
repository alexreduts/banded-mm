"""
Experimental file to try out new things

"""
from dataclasses import dataclass

import math
import numpy as np
import matplotlib.pyplot as plt

from matrix_utils import banded_matrix_generator
from matrix_mm import naive_banded_mm


#A = banded_matrix_generator(8, 3, 3)
#B = banded_matrix_generator(8, 2, 2)

#H = naive_banded_mm(A, 3, 3, B, 2, 2)
#T = np.matmul(A,B)

#print("Diff\n", H-T)
#print("H\n", H, "\nT\n", T)

#####################################################
# Naive block banded matrix multiplication
# C = alpha*A*B + beta*C
#####################################################
def blocking(matrix: np.ndarray, block_size: int):
    row, col = matrix.shape
    row_blocks, col_blocks = math.ceil(row/block_size), math.ceil(col/block_size)
    blocked_matrix = []
    
    print(matrix)
    for i in range(row_blocks):
        blocked_rows = []
        for j in range(col_blocks):
            A = matrix[
                (i*block_size):min(row,(i*block_size+block_size)),
                (j*block_size):min(col,(j*block_size+block_size))
            ]
            blocked_rows.append(A)
        blocked_matrix.append(blocked_rows)
        
    return blocked_matrix

def stacking(blocked_matrix):
    
    rows = []
    for i in range(len(blocked_matrix)):
        rows.append(np.hstack(blocked_matrix[i]))

    return np.vstack((rows))

def emtpy_block(matrix):
    empty = True
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i][j] != 0:
                empty = False
    if empty is True:
        print("multiplied empty block", )
        return True

def naive_blocked_banded_mm(
        A: np.ndarray, au: int, al: int,
        B: np.ndarray, bu: int, bl: int,
        block_size: int
        ):
    """ Naive banded matrix multiplication in blocks

    param A np.ndarray: 
    param ...

    return C np.ndarray:
    rtype: np.ndarrayd
    """

    blocked_A = blocking(A, block_size)
    blocked_B = blocking(B, block_size)
    # blocked_C = [[0 for x in range(len(blocked_A))] for x in range(len(blocked_B[0]))]
    blocked_C = blocking(np.zeros((A.shape[0], B.shape[1]), dtype=A.dtype), block_size)
    
    blocked_au = math.ceil(au/block_size)
    blocked_bu = math.ceil(bu/block_size)
    blocked_d_new = math.ceil(int((au + al + bu + bl)/2)/block_size)

    m = len(blocked_B[0])
    #print("j:", range(m))
    for j in range(m):
        #print("i:", range(max(0, j-blocked_d_new), min(m, j+blocked_d_new+1)))
        for i in range(max(0, j-blocked_d_new), min(m, j+blocked_d_new+1)):
            #print("k:", range(max(0,i-blocked_au, j-blocked_bu), min(m,i+blocked_au+1, j+blocked_bu+1)))
            for k in range(max(0,i-blocked_au, j-blocked_bu), min(m,i+blocked_au+1, j+blocked_bu+1)):
                if emtpy_block(blocked_A[i][k]):
                    print(blocked_A[i][k])
                if emtpy_block(blocked_B[k][j]):
                    print(blocked_B[k][j])
                blocked_C[i][j] += np.matmul(blocked_A[i][k],blocked_B[k][j])

    return stacking(blocked_C)

# A = banded_matrix_generator(10, 3, 3)
# B = banded_matrix_generator(10, 2, 2)
# H = naive_blocked_banded_mm(A, 3, 3, B, 2, 2, 5)
# T = np.matmul(A,B)

# This doesn't work
A = banded_matrix_generator(12, 3, 3)
B = banded_matrix_generator(12, 2, 2)
H = naive_blocked_banded_mm(A, 3, 3, B, 2, 2, 5)
T = np.matmul(A,B)

print("Diff\n", H-T)



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


