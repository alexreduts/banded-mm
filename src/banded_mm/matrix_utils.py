""" matrix_utils

Simple module containing helper functions:
- Banded matrix generator
- Binary grid matrix visualisation

"""

import numpy as np

# Banded Matrix Generator
def banded_matrix_generator(n: int, ku: int, kl: int):
    """Banded matrix generator

    Arguments:
    n: int -> matrix dimension
    ku: int -> number of subdiagonals
    kl: int -> number of superdiagonals
    """

    A = np.eye(n)

    for i in range(1,ku+1):
        A += np.eye(n, k=i)
    
    for i in range(1,kl+1):
        A += np.eye(n, k=-i)

    return A

# Check if a matrix is only zero
def empty_block(block: np.ndarray):
    """ Check if a block is empty
    """    
    ref_block = np.zeros((block.shape[0], block.shape[1]))

    return np.allclose(ref_block, block)




import matplotlib.pyplot as plt

# Binary Grid
def binary_grid(matrix: np.ndarray):
    """Binary Grid Matrix Visualization
    
    
    """
    plt.imshow(matrix)


if __name__ == "__main__":

    import sys

    dimension = sys.argv[1]
    super_diagonals = sys.argv[2]
    sub_diagonals = sys.argv[3]
 
    A = banded_matrix_generator(dimension, super_diagonals, sub_diagonals)

    binary_grid(A)