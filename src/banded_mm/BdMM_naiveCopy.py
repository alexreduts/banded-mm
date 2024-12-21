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

import numpy as np
import cupy as cp
import cupyx as cpx

# General banded times banded matrix multiplication (A & B banded)
def _gbmm_gpu_outer(
        C: np.ndarray,
        A: np.ndarray,
        ku_A: int,
        kl_A: int,
        B: np.ndarray,
        ku_B: int,
        kl_B: int
    ) -> np.ndarray:

    ku_C = ku_A + ku_B
    kl_C = kl_A + kl_B

    n = C.shape[1]

    # Tune for HW
    c = 5

    # Initialize Partition
    Cx1 = slice(0, c)
    Bx1 = slice(0, c)

    while Cx1.start < n:

        # Shrink blocks to bandwidth
        C1x = slice(max(0, Cx1.start - ku_C), min(n, Cx1.stop + kl_C))
        B1x = slice(max(0, Bx1.start - ku_B), min(n, Bx1.stop + kl_B))

        # Adjust number of upper and lower bands matching subblocks
        ku = ku_A + (C1x.start - B1x.start)
        kl = kl_A - (C1x.start - B1x.start)

        # inner loop
        C[C1x, Cx1] = _gbmm_gpu_inner(C[C1x, Cx1], A[C1x, B1x], B[B1x, Bx1], ku, kl)

        # Adjust Partition
        Cx1 = slice(Cx1.start+c, Cx1.stop+c)
        Bx1 = slice(Bx1.start+c, Bx1.stop+c)

    return C


def _gbmm_gpu_inner(
        E: np.ndarray,
        A: np.ndarray,
        D: np.ndarray,
        ku: int,
        kl: int
    ) -> np.ndarray:

    m = E.shape[0]
    k = D.shape[0]

    # Tune for HW
    b = 5

    # Buffers
    D1 = [cp.empty((b, b), dtype=A.dtype) for _ in range(2)]

    # A11 = [???]
    # A21 = [???]
    # A31 = [???]

    # E1 = [cp.empty((b, b), dtype=A.dtype) for _ in range(2)]
    # E2 = [cp.empty((b, b), dtype=A.dtype) for _ in range(2)]
    # E3 = [cp.empty((b, b), dtype=A.dtype) for _ in range(2)]

    # E = cp.empty((b,b), dtype=A.dtype) for _ in range(4)

    # Streams
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(2)]
    # Events
    events = [cp.cuda.Event() for _ in range(2)]

    # Initial Partition
    # -----------------
    D1x = slice(0, b)

    assert D1x.start < (ku+1) # Reason that this is always true ?
    E1x = slice(0, 0)

    assert D1x.start <= (k-kl-1) # Reason that this is always true ?
    E3x = slice(kl, kl+b)

    E2x = slice(E1x.stop, E3x.start)

    # Initial Calculation
    with streams[0] as stream:
        D1[0].set(D[D1x, :])
        E[E2x, :] = cp.asnumpy(cp.asarray(E[E2x, :]) + cp.matmul(cp.asarray(A[E2x, D1x]),D1[0]))
        E[E3x, :] = cp.asnumpy(cp.asarray(E[E3x, :]) + cp.matmul(cp.asarray(A[E3x, D1x]),D1[0]))
        events[0].record(stream=stream)
        # E1[0].get(out=E[E1x, :])

    # E[E2x, :] = cp.asnumpy(cp.asarray(E[E2x, :]) + cp.matmul(cp.asarray(A[E2x, D1x]),cp.asarray(D[D1x, :])))
    # E[E3x, :] = cp.asnumpy(cp.asarray(E[E3x, :]) + cp.matmul(cp.asarray(A[E3x, D1x]),cp.asarray(D[D1x, :])))

    # Adjust partition
    D1x = slice(D1x.start+b, D1x.stop+b)
    E3x = slice(E3x.start+b, E3x.stop)
    # -----------------

    # Looping
    # -----------------
    # while D1x.start < k:
    num_blocks = -(- k // b)
    for i in range(1, num_blocks):

        # Repartition
        if D1x.start < (ku+1):
            E1x = slice(E1x.start, E1x.start)
        else:
            E1x = slice(E1x.start, E1x.start+b)

        if D1x.start > (k-kl-1):
            E3x = slice(E3x.start+b, E3x.start+b)
        else:
            E3x = slice(E3x.start, E3x.start+b)

        E2x = slice(E1x.stop, E3x.start)

        # Calcuations
        with streams[i % 2] as stream:
            D1cur = D1[i % 2][:D1x.stop-D1x.start]
            D1cur.set(D[D1x, :]) 
            stream.wait_event(event=events[(i-1) % 2])

            if D1x.start >= (ku+1):
                E[E1x, :] = cp.asnumpy(cp.asarray(E[E1x, :]) + cp.matmul(cp.asarray(A[E1x, D1x]), D1cur))
                # E1[i % 2] = E1[(i-1) % 2] + A11[i % 2] @ D1[i % 2]

            if D1x.start <= (k-kl-1):
                E[E3x, :] = cp.asnumpy(cp.asarray(E[E3x, :]) + cp.matmul(cp.asarray(A[E3x, D1x]), D1cur))
                # E3[i % 2] = E3[(i-1) % 2] + A31[i % 2] @ D1[i % 2]

            E[E2x, :] = cp.asnumpy(cp.asarray(E[E2x, :]) + cp.matmul(cp.asarray(A[E2x, D1x]), D1cur))
            # E2[i % 2] = E2[(i-1) % 2] + A21[i % 2] @ D1[i % 2]
            events[i % 2].record(stream=stream)
            # E1[0].get(out=E[E1x, :])

            # Adjust partition
            if D1x.start >= (ku+1):
                E1x = slice(E1x.start+b, E1x.stop)

            # D1x = slice(D1x.start+b, D1x.stop+b)
            D1x = slice(D1x.stop, min(D1x.stop + b, k))
            E3x = slice(E3x.start+b, E3x.stop)
    # -----------------

    return E

def  gbmm_gpu(
        A: np.ndarray,
        ku_A: int,
        kl_A: int,
        B: np.ndarray,
        ku_B: int,
        kl_B: int
    ):
    C = np.zeros((A.shape[0], B.shape[1]))
    C = _gbmm_gpu_outer(C, A, ku_A, kl_A, B, ku_B, kl_B)
    return C
