""" matrix_visuals

Modul containing custom functions to for neat visualizations of matrices

"""

import numpy as np
import matplotlib.pyplot as plt

# Binary Grid
def binary_grid(matrix: np.ndarray):
    plt.imshow(matrix)