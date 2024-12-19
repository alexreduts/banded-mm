"""
Streamed implementation of xGBMM
"""

import numpy as np
import cupy as cp
import cupyx as cpx

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
        block_size_inner: int
    ) -> np.ndarray:

    ku_C = ku_A + ku_B
    kl_C = kl_A + kl_B

    n = C.shape[1]
    k = B.shape[0]

    # Cuda Streams
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(2)]
    # Cuda Events
    events = [cp.cuda.Event() for _ in range(2)]

    # Buffers
    ku, kl = ku_C, kl_C
    D1 = [cp.empty(block_size_inner * block_size_outer) for _ in range(2)]
    A11 = [cp.empty(block_size_inner * block_size_inner) for _ in range(2)]
    A21 = [cp.empty((ku+kl+block_size_inner) * block_size_inner) for _ in range(2)]
    A31 = [cp.empty(block_size_inner * block_size_inner) for _ in range(2)]
    E123 = [cp.empty((ku+kl+block_size_outer) * block_size_outer) for _ in range(2)]

    # Initialize Partition
    Cx1 = slice(0, block_size_outer)
    Bx1 = slice(0, block_size_outer)

    first_stream = 0

    while Cx1.start < n:

        # Shrink blocks to bandwidth
        C1x = slice(max(0, Cx1.start - ku_C), min(n, Cx1.stop + kl_C))
        B1x = slice(max(0, Bx1.start - ku_B), min(k, Bx1.stop + kl_B))

        # Adjust number of upper and lower bands matching subblocks
        ku = ku_A + (C1x.start - B1x.start)
        kl = kl_A - (C1x.start - B1x.start)

        # inner loop
        first_stream = _BdGEMM_inner(
            C[C1x, Cx1],
            A[C1x, B1x],
            B[B1x, Bx1],
            ku, kl,
            block_size_inner,
            streams,
            events,
            first_stream,
            D1, A11, A21, A31, E123
        )

        # Adjust Partition
        Cx1 = slice(Cx1.start+block_size_outer, Cx1.stop+block_size_outer)
        Bx1 = slice(Bx1.start+block_size_outer, Bx1.stop+block_size_outer)

    for stream in streams:
        stream.synchronize()

    return C

def _slicer(
        iter: int,
        k: int,
        m: int,
        ku: int,
        kl: int,
        block_size_inner: int
):
    """
    Explicitly calculate the borders of the blocks to be multiplied
    based on the iteration specified
    """
    
    # Position
    pos = iter*block_size_inner

    # Band limits relative to position
    band_limit_upper = max(0, pos - ku)
    band_limit_lower = min(pos + kl + block_size_inner, m)
    
    D1 = slice(pos, min(k, pos+block_size_inner))

    if pos < ku:
        A1 = slice(band_limit_upper, band_limit_upper)
    else:
        A1 = slice(band_limit_upper, min(band_limit_upper+block_size_inner, m))

    if (pos + block_size_inner) >= (m-kl-1):
        A3 = slice(band_limit_lower, band_limit_lower)
    else:
        A3 = slice(max(band_limit_lower-block_size_inner, A1.stop), band_limit_lower)

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
        first_stream,
        bD1, bA11, bA21, bA31, bE123
    ) -> np.ndarray:

    #print("A\n", A)
    #print("D\n", D)

    k = D.shape[0]
    n = D.shape[1]
    m = A.shape[0]

    # Number of blocks to compute
    num_blocks = -(-k//block_size_inner) #Rounded up

    # # Buffers
    # D1 = [cp.empty((block_size_inner, n)) for _ in range(2)]
    # # A11 = [cp.empty((block_size_inner, block_size_inner)) for _ in range(2)]
    # # A21 = [cp.empty((ku+kl+block_size_inner, block_size_inner)) for _ in range(2)]
    # # A31 = [cp.empty((block_size_inner, block_size_inner)) for _ in range(2)]
    # A11 = [cp.empty(block_size_inner * block_size_inner) for _ in range(2)]
    # A21 = [cp.empty((ku+kl+block_size_inner) * block_size_inner) for _ in range(2)]
    # A31 = [cp.empty(block_size_inner * block_size_inner) for _ in range(2)]
    # E123 = [cp.zeros((E.shape[0], E.shape[1])) for _ in range(2)]
    D1 = [bD1[0][:block_size_inner * n].reshape(block_size_inner, n), bD1[1][:block_size_inner * n].reshape(block_size_inner, n)]
    A11 = bA11
    A21 = bA21
    A31 = bA31
    E123 = [bE123[0][:E.shape[0] * E.shape[1]].reshape(E.shape[0], E.shape[1]), bE123[1][:E.shape[0] * E.shape[1]].reshape(E.shape[0], E.shape[1])]

    # Iteration step i = 0
    with streams[first_stream] as stream:

        D1_cur, A1_cur, A2_cur, A3_cur = _slicer(0, k, m, ku, kl, block_size_inner)
        D1_next, A1_next, A2_next, A3_next = _slicer(1, k, m, ku, kl, block_size_inner)

        # D1[0][:D1_cur.stop-D1_cur.start] = cp.asarray(D[D1_cur, :])
        D1[first_stream][:D1_cur.stop-D1_cur.start].set(D[D1_cur, :])

        if A1_cur.stop > A1_cur.start:
            # A11[0][:A1_cur.stop-A1_cur.start,:D1_cur.stop-D1_cur.start] = cp.asarray(A[A1_cur,D1_cur])
            tA11 = A11[first_stream][:(A1_cur.stop-A1_cur.start) * (D1_cur.stop-D1_cur.start)].reshape((A1_cur.stop-A1_cur.start), (D1_cur.stop-D1_cur.start))
            tA11.set(A[A1_cur,D1_cur])

        if A2_cur.stop > A2_cur.start:
            # A21[0][:A2_cur.stop-A2_cur.start,:D1_cur.stop-D1_cur.start] = cp.asarray(A[A2_cur,D1_cur])
            tA21 = A21[first_stream][:(A2_cur.stop-A2_cur.start) * (D1_cur.stop-D1_cur.start)].reshape((A2_cur.stop-A2_cur.start), (D1_cur.stop-D1_cur.start))
            tA21.set(A[A2_cur,D1_cur])

        if A3_cur.stop > A3_cur.start:
            # A31[0][:A3_cur.stop-A3_cur.start,:D1_cur.stop-D1_cur.start] = cp.asarray(A[A3_cur,D1_cur])
            tA31 = A31[first_stream][:(A3_cur.stop-A3_cur.start) * (D1_cur.stop-D1_cur.start)].reshape((A3_cur.stop-A3_cur.start), (D1_cur.stop-D1_cur.start))
            tA31.set(A[A3_cur,D1_cur])
        
        stream.wait_event(event=events[(first_stream + 1) % 2])
        E123[first_stream][:] = 0

        if A1_cur.stop > A1_cur.start:
            # E123[0][A1_cur, :] = E123[0][A1_cur, :] + A11[0][:A1_cur.stop-A1_cur.start,:D1_cur.stop-D1_cur.start] @ D1[0][:D1_cur.stop-D1_cur.start]
            E123[first_stream][A1_cur, :] = E123[first_stream][A1_cur, :] + tA11 @ D1[0][:D1_cur.stop-D1_cur.start]

        if A2_cur.stop > A2_cur.start:
            # E123[0][A2_cur, :] = E123[0][A2_cur, :] + A21[0][:A2_cur.stop-A2_cur.start,:D1_cur.stop-D1_cur.start] @ D1[0][:D1_cur.stop-D1_cur.start]
            E123[first_stream][A2_cur, :] = E123[first_stream][A2_cur, :] + tA21 @ D1[0][:D1_cur.stop-D1_cur.start]

        if A3_cur.stop > A3_cur.start:
            # E123[0][A3_cur, :] = E123[0][A3_cur, :] + A31[0][:A3_cur.stop-A3_cur.start,:D1_cur.stop-D1_cur.start] @ D1[0][:D1_cur.stop-D1_cur.start]
            E123[first_stream][A3_cur, :] = E123[first_stream][A3_cur, :] + tA31 @ D1[0][:D1_cur.stop-D1_cur.start]

        E123[(first_stream + 1) % 2][A2_cur.start:A3_cur.stop, :] = E123[first_stream][A2_cur.start:A3_cur.stop, :]

        events[first_stream].record(stream=stream)

        if A1_next.start > A1_cur.start:
            dst = E[A1_cur.start:A1_next.start, :].ctypes.data
            src = E123[first_stream][A1_cur.start:A1_next.start, :].data.ptr
            dpitch = E.strides[0]
            spitch = E123[first_stream].strides[0]
            width = E.shape[1] * E.itemsize
            height = A1_next.start - A1_cur.start
            kind = cp.cuda.runtime.memcpyDeviceToHost
            sptr = streams[first_stream].ptr
            cp.cuda.runtime.memcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, sptr)

        if num_blocks <= 1:
            dst = E[A1_cur.start:A3_cur.stop, :].ctypes.data
            src = E123[first_stream][A1_cur.start:A3_cur.stop, :].data.ptr
            dpitch = E.strides[0]
            spitch = E123[first_stream].strides[0]
            width = E.shape[1] * E.itemsize
            height = A3_cur.stop - A1_cur.start
            kind = cp.cuda.runtime.memcpyDeviceToHost
            sptr = streams[first_stream].ptr
            cp.cuda.runtime.memcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, sptr)

        # if A1_next.start > A1_cur.start:
        #     E[A1_cur.start:A1_next.start, :] = cp.asnumpy(E123[0][A1_cur.start:A1_next.start, :])
        #     # E123[0][A1_cur.start:A1_next.start, :].get(out=E[A1_cur.start:A1_next.start, :])
        # if num_blocks <= 1:
        #     E[A1_cur.start:A3_cur.stop, :] = cp.asnumpy(E123[0][A1_cur.start:A3_cur.stop, :])
        #     # E123[0][A1_cur.start:A3_cur.stop, :].get(out=E[A1_cur.start:A3_cur.stop, :])    


    # Iteration step i = 1 to i = num_blocks-1
    for i in range(1, num_blocks):
        
        with streams[(first_stream + i) % 2] as stream:

            D1_cur, A1_cur, A2_cur, A3_cur = _slicer(i, k, m, ku, kl, block_size_inner)
            D1_next, A1_next, A2_next, A3_next = _slicer(i+1, k, m, ku, kl, block_size_inner)

            # D1[i%2][:D1_cur.stop-D1_cur.start] = cp.asarray(D[D1_cur, :])
            D1[(first_stream + i) % 2][:D1_cur.stop-D1_cur.start].set(D[D1_cur, :])

            if A1_cur.stop > A1_cur.start:
                # A11[i%2][:A1_cur.stop-A1_cur.start,:D1_cur.stop-D1_cur.start] = cp.asarray(A[A1_cur,D1_cur])
                tA11 = A11[(first_stream + i) % 2][:(A1_cur.stop-A1_cur.start) * (D1_cur.stop-D1_cur.start)].reshape((A1_cur.stop-A1_cur.start), (D1_cur.stop-D1_cur.start))
                tA11.set(A[A1_cur,D1_cur])

            if A2_cur.stop > A2_cur.start:
                # A21[i%2][:A2_cur.stop-A2_cur.start,:D1_cur.stop-D1_cur.start] = cp.asarray(A[A2_cur,D1_cur])
                tA21 = A21[(first_stream + i) % 2][:(A2_cur.stop-A2_cur.start) * (D1_cur.stop-D1_cur.start)].reshape((A2_cur.stop-A2_cur.start), (D1_cur.stop-D1_cur.start))
                tA21.set(A[A2_cur,D1_cur])

            if A3_cur.stop > A3_cur.start:
                # A31[i%2][:A3_cur.stop-A3_cur.start,:D1_cur.stop-D1_cur.start] = cp.asarray(A[A3_cur,D1_cur])
                tA31 = A31[(first_stream + i) % 2][:(A3_cur.stop-A3_cur.start) * (D1_cur.stop-D1_cur.start)].reshape((A3_cur.stop-A3_cur.start), (D1_cur.stop-D1_cur.start))
                tA31.set(A[A3_cur,D1_cur])
            
            if i == 1:
                E123[(first_stream + i) % 2][:] = 0

            stream.wait_event(event=events[(first_stream + i - 1) % 2])

            if A1_cur.stop > A1_cur.start:
                # E123[i%2][A1_cur, :] = E123[i%2][A1_cur, :] + A11[i%2][:A1_cur.stop-A1_cur.start,:D1_cur.stop-D1_cur.start] @ D1[i%2][:D1_cur.stop-D1_cur.start]
                E123[(first_stream + i) % 2][A1_cur, :] = E123[(first_stream + i) % 2][A1_cur, :] + tA11 @ D1[(first_stream + i) % 2][:D1_cur.stop-D1_cur.start]

            if A2_cur.stop > A2_cur.start:
                # E123[i%2][A2_cur, :] = E123[i%2][A2_cur, :] + A21[i%2][:A2_cur.stop-A2_cur.start,:D1_cur.stop-D1_cur.start] @ D1[i%2][:D1_cur.stop-D1_cur.start]
                E123[(first_stream + i) % 2][A2_cur, :] = E123[(first_stream + i) % 2][A2_cur, :] + tA21 @ D1[(first_stream + i) % 2][:D1_cur.stop-D1_cur.start]

            if A3_cur.stop > A3_cur.start:
                # E123[i%2][A3_cur, :] = E123[i%2][A3_cur, :] + A31[i%2][:A3_cur.stop-A3_cur.start,:D1_cur.stop-D1_cur.start] @ D1[i%2][:D1_cur.stop-D1_cur.start]
                E123[(first_stream + i) % 2][A3_cur, :] = E123[(first_stream + i) % 2][A3_cur, :] + tA31 @ D1[(first_stream + i) % 2][:D1_cur.stop-D1_cur.start]

            E123[(first_stream + i + 1) % 2][A2_cur.start:A3_cur.stop, :] = E123[(first_stream + i) % 2][A2_cur.start:A3_cur.stop, :]

            events[(first_stream + i) % 2].record(stream=stream)

            if A1_next.start > A1_cur.start:
                dst = E[A1_cur.start:A1_next.start, :].ctypes.data
                src = E123[(first_stream + i) % 2][A1_cur.start:A1_next.start, :].data.ptr
                dpitch = E.strides[0]
                spitch = E123[(first_stream + i) % 2].strides[0]
                width = E.shape[1] * E.itemsize
                height = A1_next.start - A1_cur.start
                kind = cp.cuda.runtime.memcpyDeviceToHost
                sptr = streams[(first_stream + i) % 2].ptr
                cp.cuda.runtime.memcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, sptr)

            if i >= (num_blocks-1):
                dst = E[A1_cur.start:A3_cur.stop, :].ctypes.data
                src = E123[(first_stream + i) % 2][A1_cur.start:A3_cur.stop, :].data.ptr
                dpitch = E.strides[0]
                spitch = E123[(first_stream + i) % 2].strides[0]
                width = E.shape[1] * E.itemsize
                height = A3_cur.stop - A1_cur.start
                kind = cp.cuda.runtime.memcpyDeviceToHost
                sptr = streams[(first_stream + i) % 2].ptr
                cp.cuda.runtime.memcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, sptr)

            # if A1_next.start > A1_cur.start:
            #     E[A1_cur.start:A1_next.start, :] = cp.asnumpy(E123[i%2][A1_cur.start:A1_next.start, :])

            # if i >= (num_blocks-1):
            #     E[A1_cur.start:A3_cur.stop, :] = cp.asnumpy(E123[i%2][A1_cur.start:A3_cur.stop, :])
    
    # for stream in streams:
    #     stream.synchronize()

    return (first_stream + num_blocks) % 2

def  BdGEMM_streamed(
        C: np.ndarray,
        A: np.ndarray,
        kl_A: int,
        ku_A: int,
        B: np.ndarray,
        kl_B: int,
        ku_B: int,
        block_size_outer,
        block_size_inner
    ):
    # C = np.zeros((A.shape[0], B.shape[1]))
    C = _BdGEMM_outer(C, A, ku_A, kl_A, B, ku_B, kl_B, block_size_outer, block_size_inner)
    return C
