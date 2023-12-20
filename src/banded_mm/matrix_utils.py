""" matrix_utils

Simple module containing helper functions:
- Banded matrix generator
- Binary grid matrix visualisation

"""

import numpy as np

# Banded Matrix Generator
def banded_matrix_generator(m: int, n: int, kl: int, ku: int):
    """Banded matrix generator

    Arguments:
    m: int -> matrix rows
    n: int -> matrix columns
    ku: int -> number of upper band diagonals
    kl: int -> number of lower band diagonals
    """
    
    if ku < 0 or kl < 0:
        raise ValueError("Bandwidths must be non-negative")

    if m <= 0 and n <= 0:
        raise ValueError("Matrix size must be positive")

    matrix = np.zeros((m, n))

    rows, cols = np.indices((m, n))
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

    rows = int(sys.argv[1])
    columns = int(sys.argv[2])
    lower_diagonals = int(sys.argv[3])
    upper_diagonals = int(sys.argv[4])
 
    A = banded_matrix_generator(rows, columns, lower_diagonals, upper_diagonals)

    print(A)
    binary_grid(A)