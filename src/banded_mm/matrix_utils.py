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
    ku: int -> number of upper band diagonals
    kl: int -> number of lower band diagonals
    """
    
    if ku < 0 or kl < 0:
        raise ValueError("Bandwidths must be non-negative")

    if n <= 0:
        raise ValueError("Matrix size must be positive")

    matrix = np.zeros((n, n))

    rows, cols = np.indices((n, n))
    mask = (cols >= rows - kl) & (cols <= rows + ku)
    
    matrix[mask] = 1

    return matrix

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

    dimension = int(sys.argv[1])
    super_diagonals = int(sys.argv[2])
    sub_diagonals = int(sys.argv[3])
 
    A = banded_matrix_generator(dimension, super_diagonals, sub_diagonals)

    binary_grid(A)