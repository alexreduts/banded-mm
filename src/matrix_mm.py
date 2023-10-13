"""matrix_mm

Modul containing different matrix multiplication algorithms:
- naive dense matrix multiplication algorithm
- Strassen matrix multiplication algorithm

"""

from time import time

import numpy as np


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
