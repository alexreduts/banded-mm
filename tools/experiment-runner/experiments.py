###
# Experiments
###

from timeit import repeat
import numpy as np

## Parse input
match test_scenario:
    case BdGEMM_blocking:
        


# --------------------------------
### Moved from BdGEMM_blocking.py
from matrix_utils import banded_matrix_generator

if __name__ == "__main__":
    import sys
    import time

    flagged = False
    try:
        if sys.argv[1] == "--profiling":
            flagged = True
    except:
        pass

    if flagged:
        print("Profiling Setup Used")
        print("Generating band matrices")
        A = banded_matrix_generator(10000, 10000, 2400, 2900)
        B = banded_matrix_generator(10000, 10000, 3000, 800)

        print("Calculating xGBMM_streamed")
        total_time = 0
        for i in range(10):
            start_time = time.time()
            C = BdGEMM_blocking(A, 2400, 2900, B, 3000, 800, 3000, 3000)
            end_time = time.time()
            total_time += end_time - start_time
        print(f"Average Time taken: {total_time/10} seconds")
    else:
        print("Debug Setup Used")
        print("Generating band matrices")
        # A = banded_matrix_generator(10, 2, 2)
        # B = banded_matrix_generator(10, 0, 2)
        A = banded_matrix_generator(10, 10, 2, 2)
        B = banded_matrix_generator(10, 10, 2, 0)

        print("Calculating xGBMM")
        # C = gbmm_gpu(A, 2, 2, B, 0, 2, 3, 2)
        C = BdGEMM_blocking(A, 2, 2, B, 2, 0, 3, 2)

    print("Calculating Ref with numpy")
    # total_time = 0
    # for i in range(10):
    #    start_time = time.time()
    #    T = A @ B
    #    end_time = time.time()
    #    total_time += end_time - start_time
    # print(f"Average Time taken: {total_time/10} seconds")
    T = A @ B

    # print("Calculating xGBMM_naive_copy")
    # total_time = 0
    # for i in range(10):
    #    start_time = time.time()
    #    C = xGBMM_naive_copy(A, 2400, 2900, B, 3000, 800, 3000, 3000)
    #    end_time = time.time()
    #    total_time += end_time - start_time
    # print(f"Average Time taken: {total_time/10} seconds")

    # print(T)
    # print(C)
    assert np.allclose(C, T)
    print("Correct Result computed")

### Moved from BdGEMM_blocking.py
# ---------------------------------

# ---------------------------------
### Moved from BdGEMM_naiveCopy.py
if __name__ == "__main__":

    import sys

    flagged = False
    try:
        if sys.argv[1] == "--profiling":
            flagged = True
    except:
        pass

    if flagged:
        print("Profiling Setup Used")
        print("Generating band matrices")
        A = banded_matrix_generator(10000, 10000, 2400, 2900)
        B = banded_matrix_generator(10000, 10000, 3000, 800)

        print("Calculating xGBMM")
        C = xGBMM_naive_copy(A, 2400, 2900, B, 3000, 800, 3000, 3000)
    else:
        print("Debug Setup Used")
        print("Generating band matrices")
        # A = banded_matrix_generator(10, 10, 0, 0)
        # B = banded_matrix_generator(10, 10, 0, 0)
        A = banded_matrix_generator(5, 10, 2, 3)
        B = banded_matrix_generator(10, 5, 2, 3)

        print("Calculating xGBMM")
        # C = xGBMM_naive_copy(A, 0, 0, B, 0, 0, 3, 2)
        C = xGBMM_naive_copy(A, 2, 3, B, 2, 3, 3, 2)

    print("Calculating Ref with numpy")
    T = A @ B
    assert np.allclose(C, T)
    print("Correct Result computed")

### Moved from BdGEMM_naiveCopy.py
# ---------------------------------

# ---------------------------------
### Moved from BdGEMM_streamed.py
if __name__ == "__main__":

    import sys
    import time

    flagged = True
    try:
        if sys.argv[1] == "--profiling":
            flagged = True
    except:
        pass

    if flagged:
        print("Profiling Setup Used")
        print("Generating band matrices")
        A = banded_matrix_generator(2000, 2000, 50, 50)
        B = banded_matrix_generator(2000, 2000, 50, 50)
        C = cpx.zeros_pinned((A.shape[0], B.shape[1]))

        # print("Calculating xGBMM_streamed")
        # total_time = 0
        # for i in range(10):
        #     start_time = time.time()
        #     C = xGBMM_streamed(C, A, 50, 50, B, 50, 50, 100, 100)
        #     end_time = time.time()
        #     total_time += end_time - start_time
        # print(f"Average Time taken: {total_time/10} seconds")

        runtimes = repeat(
            "BdGEMM_streamed(C, A, 50, 50, B, 50, 50, 100, 100)",
            setup="cp.cuda.runtime.deviceSynchronize()",
            repeat=20,
            number=1,
            globals={**globals(), **locals()},
        )
        print(f"Median Time taken: {np.median(runtimes)} seconds")

        A2 = cp.sparse.csr_matrix(cp.asarray(A))
        B2 = cp.sparse.csr_matrix(cp.asarray(B))
        C2 = A2 @ B2
        print(type(C2))
        C3 = C2.toarray()
        assert np.allclose(C, C3)
        runtimes = repeat(
            "A2 @ B2",
            setup="cp.cuda.runtime.deviceSynchronize()",
            repeat=20,
            number=1,
            globals={**globals(), **locals()},
        )
        print(f"Median Time taken: {np.median(runtimes)} seconds")

    else:
        print("Debug Setup Used")
        print("Generating band matrices")
        # A = banded_matrix_generator(10, 2, 2)
        # B = banded_matrix_generator(10, 0, 2)
        A = banded_matrix_generator(10, 10, 2, 2)
        B = banded_matrix_generator(10, 10, 2, 0)
        C = cpx.zeros_pinned((A.shape[0], B.shape[1]))

        print("Calculating xGBMM")
        # C = gbmm_gpu(A, 2, 2, B, 0, 2, 3, 2)
        C = BdGEMM_streamed(C, A, 2, 2, B, 2, 0, 3, 2)

    print("Calculating Ref with numpy")
    # total_time = 0
    # for i in range(10):
    #    start_time = time.time()
    #    T = A @ B
    #    end_time = time.time()
    #    total_time += end_time - start_time
    # print(f"Average Time taken: {total_time/10} seconds")
    T = A @ B

    # print("Calculating xGBMM_naive_copy")
    # total_time = 0
    # for i in range(10):
    #    start_time = time.time()
    #    C = xGBMM_naive_copy(A, 2400, 2900, B, 3000, 800, 3000, 3000)
    #    end_time = time.time()
    #    total_time += end_time - start_time
    # print(f"Average Time taken: {total_time/10} seconds")

    # print(T)
    # print(C)
    print(np.linalg.norm(C - T) / np.linalg.norm(T))
    assert np.allclose(C, T)
    print("Correct Result computed")

# --------------------------------
### Moved from BdGEMM_streamed.py

# -------------------------------
### Moved from BdMM_naiveCopy.py

if __name__ == "__main__":

    A = banded_matrix_generator(25, 2, 1)
    B = banded_matrix_generator(25, 3, 7)
    C = gbmm_gpu(A, 2, 1, B, 3, 7)

    T = A @ B
    # print(C-T)
    assert np.allclose(C, T)
    print("Correct Result computed")

### moved from BdMM_naiveCopy.py
# -------------------------------
