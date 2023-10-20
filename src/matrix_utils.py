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

    rng = np.random.default_rng(seed=42)
    A = np.diag(rng.random(n))
    
    for i in range(1,ku+1):
        A += np.diag(rng.random(n-i), k=i)
    
    for i in range(1,kl+1):
        A += np.diag(rng.random(n-i), k=-i)

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
