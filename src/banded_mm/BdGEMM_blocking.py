"""
Copyright [2024] [Alex Studer]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Blocking implementation of BdGEMM
"""

import numpy as np
import cupy as cp

# General banded times banded matrix multiplication (A & B banded)
def _BdGEMM_outer(
    C: np.ndarray,
    A: np.ndarray,
    ku_A: int,
    kl_A: int,
    B: np.ndarray,
    ku_B: int,
    kl_B: int,
    block_size_outer: int,
    block_size_inner: int,
) -> np.ndarray:
    ku_C = ku_A + ku_B
    kl_C = kl_A + kl_B

    n = C.shape[1]
    k = B.shape[0]

    # Cuda Streams
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(2)]
    # Cuda Events
    events = [cp.cuda.Event() for _ in range(2)]

    # Initialize Partition
    Cx1 = slice(0, block_size_outer)
    Bx1 = slice(0, block_size_outer)

    while Cx1.start < n:
        # Shrink blocks to bandwidth
        C1x = slice(max(0, Cx1.start - ku_C), min(n, Cx1.stop + kl_C))
        B1x = slice(max(0, Bx1.start - ku_B), min(k, Bx1.stop + kl_B))

        # Adjust number of upper and lower bands matching subblocks
        ku = ku_A + (C1x.start - B1x.start)
        kl = kl_A - (C1x.start - B1x.start)

        # inner loop
        C[C1x, Cx1] = _BdGEMM_inner(
            C[C1x, Cx1],
            A[C1x, B1x],
            B[B1x, Bx1],
            ku,
            kl,
            block_size_inner,
            streams,
            events,
        )

        # Adjust Partition
        Cx1 = slice(Cx1.start + block_size_outer, Cx1.stop + block_size_outer)
        Bx1 = slice(Bx1.start + block_size_outer, Bx1.stop + block_size_outer)

    return C


def _slicer(iter: int, k: int, m: int, ku: int, kl: int, block_size_inner: int):
    """
    Explicitly calculate the borders of the blocks to be multiplied
    based on the iteration specified
    """

    # Position
    pos = iter * block_size_inner

    # Band limits relative to position
    band_limit_upper = max(0, pos - ku)
    band_limit_lower = min(pos + kl + block_size_inner, m)

    D1 = slice(pos, min(k, pos + block_size_inner))

    if pos < ku:
        A1 = slice(band_limit_upper, band_limit_upper)
    else:
        A1 = slice(band_limit_upper, min(band_limit_upper + block_size_inner, m))

    if (pos + block_size_inner) >= (m - kl - 1):
        A3 = slice(band_limit_lower, band_limit_lower)
    else:
        A3 = slice(max(band_limit_lower - block_size_inner, A1.stop), band_limit_lower)

    A2 = slice(A1.stop, A3.start)

    return D1, A1, A2, A3


def _BdGEMM_inner(
    E: np.ndarray,
    A: np.ndarray,
    D: np.ndarray,
    ku: int,
    kl: int,
    block_size_inner,
    streams,
    events,
) -> np.ndarray:
    # print("A\n", A)
    # print("D\n", D)

    k = D.shape[0]
    n = D.shape[1]
    m = A.shape[0]

    # Number of blocks to compute
    num_blocks = -(-k // block_size_inner)  # Rounded up

    # Buffers
    D1 = [cp.empty((block_size_inner, n)) for _ in range(2)]

    A11 = [cp.empty((block_size_inner, block_size_inner)) for _ in range(2)]
    A21 = [cp.empty((ku + kl + block_size_inner, block_size_inner)) for _ in range(2)]
    A31 = [cp.empty((block_size_inner, block_size_inner)) for _ in range(2)]

    E123 = [cp.zeros((E.shape[0], E.shape[1])) for _ in range(2)]

    # Iteration step i = 0
    with streams[0] as stream:
        D1_cur, A1_cur, A2_cur, A3_cur = _slicer(0, k, m, ku, kl, block_size_inner)
        D1_next, A1_next, A2_next, A3_next = _slicer(1, k, m, ku, kl, block_size_inner)

        D1[0][: D1_cur.stop - D1_cur.start] = cp.asarray(D[D1_cur, :])

        if A1_cur.stop > A1_cur.start:
            A11[0][
                : A1_cur.stop - A1_cur.start, : D1_cur.stop - D1_cur.start
            ] = cp.asarray(A[A1_cur, D1_cur])

        if A2_cur.stop > A2_cur.start:
            A21[0][
                : A2_cur.stop - A2_cur.start, : D1_cur.stop - D1_cur.start
            ] = cp.asarray(A[A2_cur, D1_cur])

        if A3_cur.stop > A3_cur.start:
            A31[0][
                : A3_cur.stop - A3_cur.start, : D1_cur.stop - D1_cur.start
            ] = cp.asarray(A[A3_cur, D1_cur])

        if A1_cur.stop > A1_cur.start:
            E123[0][A1_cur, :] = (
                E123[0][A1_cur, :]
                + A11[0][: A1_cur.stop - A1_cur.start, : D1_cur.stop - D1_cur.start]
                @ D1[0][: D1_cur.stop - D1_cur.start]
            )

        if A2_cur.stop > A2_cur.start:
            E123[0][A2_cur, :] = (
                E123[0][A2_cur, :]
                + A21[0][: A2_cur.stop - A2_cur.start, : D1_cur.stop - D1_cur.start]
                @ D1[0][: D1_cur.stop - D1_cur.start]
            )

        if A3_cur.stop > A3_cur.start:
            E123[0][A3_cur, :] = (
                E123[0][A3_cur, :]
                + A31[0][: A3_cur.stop - A3_cur.start, : D1_cur.stop - D1_cur.start]
                @ D1[0][: D1_cur.stop - D1_cur.start]
            )

        E123[1][A2_cur.start : A3_cur.stop, :] = E123[0][A2_cur.start : A3_cur.stop, :]

        events[0].record(stream=stream)

        if A1_next.start > A1_cur.start:
            E[A1_cur.start : A1_next.start, :] = cp.asnumpy(
                E123[0][A1_cur.start : A1_next.start, :]
            )
        if num_blocks <= 1:
            E[A1_cur.start : A3_cur.stop, :] = cp.asnumpy(
                E123[0][A1_cur.start : A3_cur.stop, :]
            )

    # Iteration step i = 1 to i = num_blocks-1
    for i in range(1, num_blocks):
        with streams[i % 2] as stream:
            D1_cur, A1_cur, A2_cur, A3_cur = _slicer(i, k, m, ku, kl, block_size_inner)
            D1_next, A1_next, A2_next, A3_next = _slicer(
                i + 1, k, m, ku, kl, block_size_inner
            )

            D1[i % 2][: D1_cur.stop - D1_cur.start] = cp.asarray(D[D1_cur, :])

            if A1_cur.stop > A1_cur.start:
                A11[i % 2][
                    : A1_cur.stop - A1_cur.start, : D1_cur.stop - D1_cur.start
                ] = cp.asarray(A[A1_cur, D1_cur])

            if A2_cur.stop > A2_cur.start:
                A21[i % 2][
                    : A2_cur.stop - A2_cur.start, : D1_cur.stop - D1_cur.start
                ] = cp.asarray(A[A2_cur, D1_cur])

            if A3_cur.stop > A3_cur.start:
                A31[i % 2][
                    : A3_cur.stop - A3_cur.start, : D1_cur.stop - D1_cur.start
                ] = cp.asarray(A[A3_cur, D1_cur])

            stream.wait_event(event=events[(i - 1) % 2])

            if A1_cur.stop > A1_cur.start:
                E123[i % 2][A1_cur, :] = (
                    E123[i % 2][A1_cur, :]
                    + A11[i % 2][
                        : A1_cur.stop - A1_cur.start, : D1_cur.stop - D1_cur.start
                    ]
                    @ D1[i % 2][: D1_cur.stop - D1_cur.start]
                )

            if A2_cur.stop > A2_cur.start:
                E123[i % 2][A2_cur, :] = (
                    E123[i % 2][A2_cur, :]
                    + A21[i % 2][
                        : A2_cur.stop - A2_cur.start, : D1_cur.stop - D1_cur.start
                    ]
                    @ D1[i % 2][: D1_cur.stop - D1_cur.start]
                )

            if A3_cur.stop > A3_cur.start:
                E123[i % 2][A3_cur, :] = (
                    E123[i % 2][A3_cur, :]
                    + A31[i % 2][
                        : A3_cur.stop - A3_cur.start, : D1_cur.stop - D1_cur.start
                    ]
                    @ D1[i % 2][: D1_cur.stop - D1_cur.start]
                )

            E123[(i + 1) % 2][A2_cur.start : A3_cur.stop, :] = E123[i % 2][
                A2_cur.start : A3_cur.stop, :
            ]

            events[i % 2].record(stream=stream)

            if A1_next.start > A1_cur.start:
                E[A1_cur.start : A1_next.start, :] = cp.asnumpy(
                    E123[i % 2][A1_cur.start : A1_next.start, :]
                )

            if i >= (num_blocks - 1):
                E[A1_cur.start : A3_cur.stop, :] = cp.asnumpy(
                    E123[i % 2][A1_cur.start : A3_cur.stop, :]
                )

    return E


def BdGEMM_blocking(
    A: np.ndarray,
    kl_A: int,
    ku_A: int,
    B: np.ndarray,
    kl_B: int,
    ku_B: int,
    block_size_outer,
    block_size_inner,
):
    C = np.zeros((A.shape[0], B.shape[1]))
    C = _BdGEMM_outer(C, A, ku_A, kl_A, B, ku_B, kl_B, block_size_outer, block_size_inner)
    return C

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