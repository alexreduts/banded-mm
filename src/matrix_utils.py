""" matrix_utils

Simple module containing helper functions:
- Banded matrix generator

"""

import numpy as np


def banded_matric_generator(n: int, ku: int, kl: int):
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
