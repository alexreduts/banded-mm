# Imports
import numpy as np
#from matrix_visuals import binary_grid

# Banded matrix_generator
def banded_matric_generator(n: int, ku: int, kl: int):
    rng = np.random.default_rng(seed=42)
    A = np.diag(rng.random(n))
    
    for i in range(1,ku+1):
        A += np.diag(rng.random(n-i), k=i)
    
    for i in range(1,kl+1):
        A += np.diag(rng.random(n-i), k=-i)

    print("Generatred Banded Matrix:\n", A)
    return A

A = banded_matric_generator(8, 1, 1)
B = banded_matric_generator(8, 1, 1)
#binary_grid(A)
print(np.matmul(A,B))
