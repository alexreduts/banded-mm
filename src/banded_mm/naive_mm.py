"""matrix_mm

Modul containing different matrix multiplication algorithms:
- Naive dense matrix multiplication
- Naive blocked dense matrix multiplication
- Strassen matrix multiplication
- Naive banded matrix multiplication
- Naive explicitly blocked gbmm
- Naive implicitly blocked gbmm

"""

import numpy as np

from banded_mm.matrix_utils import empty_block

# Naive (Textbook) dense matrix multiplication (3-loops)
def naive_dense_mm(A, B):
    """Textbook (naive) dense matrix multiplication algorithm

    Input: nxn matrices A and B
    Output: nxn matrix, product of A and B

    """

    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                C[i,j] += A[i,k] * B[k,j]

# Naive blocked dense matrix multiplication
def naive_blocked_mm(
        A: np.ndarray,
        B: np.ndarray,
        block_size: int
        ):
    """ Naive banded matrix multiplication in blocks

    param A np.ndarray: 
    param ...

    return C np.ndarray:
    rtype: np.ndarrayd
    """

    C = np.zeros((A.shape[0], B.shape[1]), dtype=A.dtype)

    for i in range(C.shape[0]):
        i_ = slice(i*block_size, min(C.shape[0], (i+1)*block_size))
        for j in range(C.shape[1]):
            j_ = slice(j*block_size, min(C.shape[1], (j+1)*block_size))
            for k in range(A.shape[1]):
                k_ = slice(k*block_size, min(A.shape[1], (k+1)*block_size))
                C[i_, j_] += np.matmul(A[i_, k_], B[k_, j_])

    return C


# Strassen Matrix Multiplication Algorithm
def _strassen_split(matrix):
    """
    Splits a given matrix into quarters.
    Input: nxn matrix
    Output: tuple containing 4 n/2 x n/2 matrices corresponding to a, b, c, d
    """
    row, col = matrix.shape
    row2, col2 = row//2, col//2
    return matrix[:row2, :col2], matrix[:row2, col2:], matrix[row2:, :col2], matrix[row2:, col2:]


def strassen(A, B):
    """ 
    Computes matrix product by divide and conquer approach, recursively.
    Input: nxn matrices A and B
    Output: nxn matrix, product of A and B
    """
 
    # Base case when size of matrices is 1x1
    if len(A) == 1:
        return A * B
 
    # Splitting the matrices into quadrants. This will be done recursively
    # until the base case is reached.
    a, b, c, d = _strassen_split(A)
    e, f, g, h = _strassen_split(B)
 
    # Computing the 7 products, recursively (p1, p2...p7)
    p1 = strassen(a, f - h) 
    p2 = strassen(a + b, h)       
    p3 = strassen(c + d, e)       
    p4 = strassen(d, g - e)       
    p5 = strassen(a + d, e + h)       
    p6 = strassen(b - d, g + h) 
    p7 = strassen(a - c, e + f) 
 
    # Computing the values of the 4 quadrants of the final matrix c
    c11 = p5 + p4 - p2 + p6 
    c12 = p1 + p2          
    c21 = p3 + p4           
    c22 = p1 + p5 - p3 - p7 
 
    # Combining the 4 quadrants into a single matrix by stacking horizontally and vertically.
    C = np.vstack((np.hstack((c11, c12)), np.hstack((c21, c22))))
 
    return C

# Naive banded matrix multiplication
def naive_banded_mm(A, au, al, B, bu, bl):
    """ Naive banded matrix multiplication algorithm

    :param np.ndarray A: nxn matrix
    :param int au: # of superdiagonals
    :param int al: # of subdiagonals
    :param np.ndarray B: nxn matrix
    :param int bu: # of superdiagonals
    :param int bl: # of subdiagonals

    :rtype: np.ndarray
    :returns: matrix product of A*B
    """
    
    #symmetric bandwidth
    
    d_new = int((au + al + bu + bl)/2)
    n = A.shape[0]
    m = B.shape[1]

    C = np.zeros((n, m))
    
    count_j = 0
    count_i = 0
    count_k = 0
    count_A_zeros = 0
    count_B_zeros = 0
   
    #print("j: ", range(m))
    for j in range(m):
        count_j += 1

        #print("i: ", range(max(0, j-d_new), min(m, j+d_new+1)))
        for i in range(max(0, j-d_new), min(m, j+d_new+1)):    
            count_i += 1

            #print("k: ", range(max(0, i-au, j-bu), min(m, i+au+1, j+bu+1)))
            for k in range(max(0,i-au, j-bu), min(m,i+au+1, j+bu+1)):
                count_k += 1
                
                if(A[i,k] == 0):
                    count_A_zeros += 1
                if(B[k,j] == 0):
                    count_B_zeros += 1

                C[i,j] += A[i,k] * B[k,j]
    
    print("A zeros: ", count_A_zeros, "B zeros: ", count_B_zeros)
    print("count_j: ", count_j, " count_i: ", count_i, " count_k: ", count_k)
    return C


# Naive EXPLICITLY blocked gbmm
def blocking(matrix: np.ndarray, block_size: int):
    row, col = matrix.shape
    row_blocks, col_blocks = -(-row//block_size), -(-col//block_size)
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



def explicit_blocked_banded_mm(
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
    
    blocked_au = -(-au//block_size) # ceiling(x/y) = -(-x//y)
    blocked_bu = -(-bu//block_size)
    blocked_d_new = -(-int((au + al + bu + bl)/2)//block_size)

    m = len(blocked_B[0])
    #print("j:", range(m))
    for j in range(m):
        #print("i:", range(max(0, j-blocked_d_new), min(m, j+blocked_d_new+1)))
        for i in range(max(0, j-blocked_d_new), min(m, j+blocked_d_new+1)):
            #print("k:", range(max(0,i-blocked_au, j-blocked_bu), min(m,i+blocked_au+1, j+blocked_bu+1)))
            for k in range(max(0,i-blocked_au, j-blocked_bu), min(m,i+blocked_au+1, j+blocked_bu+1)):
                if empty_block(blocked_A[i][k]):
                    #print(blocked_A[i][k])
                    pass
                if empty_block(blocked_B[k][j]):
                    #print(blocked_B[k][j])
                    pass
                blocked_C[i][j] += np.matmul(blocked_A[i][k],blocked_B[k][j])

    return stacking(blocked_C)


# Naive IMPLICITLY blocked gbmm
def naive_blocked_banded_mm(
        A: np.ndarray, nnz_A: int,
        B: np.ndarray, nnz_B: int,
        block_size: int
        ):
    """ Naive banded matrix multiplication in blocks

    param A np.ndarray: 
    param ...

    return C np.ndarray:
    rtype: np.ndarrayd
    """

    #nnz = number of non-zero per row/column = bandwidth
    nnz_C = int((nnz_A-1))+int((nnz_B-1))+1
    #s_diag = super/sub diagonals
    s_diag_C = int((nnz_C-1)/2) #((nnz_A-1)/2)+((nnz_B-1)/2)
    #b_diag = block super/sub diagonals
    b_diag_C = -(-s_diag_C//block_size)
    print("nnz_C: ", nnz_C, " s_diag_C: ", s_diag_C, " b_diag_C: ", b_diag_C)

    C = np.zeros((A.shape[0], B.shape[1]), dtype=A.dtype)
    #print("A:", A)
    #print("B:", B)
    #print("C:", C)
    empty_blocks = 0
    for i in range(-(-C.shape[0]//block_size)):
        i_ = slice(i*block_size, min(C.shape[0], (i+1)*block_size))
        #print("i_:", slice(i*block_size, min(C.shape[0], (i+1)*block_size)))
        for j in range(max(0, i-b_diag_C), min(i+b_diag_C+1, -(-C.shape[1]//block_size))):
            j_ = slice(j*block_size, min(C.shape[1], (j+1)*block_size))
            #print("j_:", slice(j*block_size, min(C.shape[1], (j+1)*block_size)))
            for k in range(max(0, i-int((nnz_A-1)/2), j-int((nnz_B-1)/2)), min(i+int((nnz_A-1)/2)+1, j+int((nnz_B-1)/2)+1, -(-A.shape[1]//block_size))):
                k_ = slice(k*block_size, min(A.shape[1], (k+1)*block_size))
                #print("k_:", slice(k*block_size, min(A.shape[1], (k+1)*block_size)))
                flag = False
                if empty_block(A[i_, k_]) or empty_block(B[k_, j_]):
                    print("i_: ", i_, "k_: ", k_)
                    empty_blocks += 1
                    print("k_: ", k_, "j_: ", j_)
                    flag = True
                    empty_blocks += 1
                if flag: print("------")

                C[i_, j_] += np.matmul(A[i_, k_], B[k_, j_])

    
    return C