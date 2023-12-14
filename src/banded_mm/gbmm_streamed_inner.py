#b
import numpy as np
import cupy as cp

from banded_mm.matrix_utils import banded_matrix_generator

# General banded times banded matrix multiplication (A & B banded)
def _gbmm_gpu_outer(
        C: np.ndarray,
        A: np.ndarray,
        ku_A: int,
        kl_A: int,
        B: np.ndarray,
        ku_B: int,
        kl_B: int,
        block_size_inner: int,
        block_size_outer: int
    ) -> np.ndarray:

    ku_C = ku_A + ku_B
    kl_C = kl_A + kl_B

    n = C.shape[1]

    # Cuda Streams
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(2)]
    # Cuda Events
    events = [cp.cuda.Event() for _ in range(4)]

    # Initialize Partition
    Cx1 = slice(0, block_size_outer)
    Bx1 = slice(0, block_size_outer)

    while Cx1.start < n:

        # Shrink blocks to bandwidth
        C1x = slice(max(0, Cx1.start - ku_C), min(n, Cx1.stop + kl_C))
        B1x = slice(max(0, Bx1.start - ku_B), min(n, Bx1.stop + kl_B))

        # Adjust number of upper and lower bands matching subblocks
        ku = ku_A + (C1x.start - B1x.start)
        kl = kl_A - (C1x.start - B1x.start)

        # inner loop
        C[C1x, Cx1] = _gbmm_gpu_inner(
            C[C1x, Cx1],
            A[C1x, B1x],
            B[B1x, Bx1],
            ku, kl,
            block_size_inner,
            streams,
            events
        )

        # Adjust Partition
        Cx1 = slice(Cx1.start+block_size_outer, Cx1.stop+block_size_outer)
        Bx1 = slice(Bx1.start+block_size_outer, Bx1.stop+block_size_outer)

    return C

def _slicer(
        iter: int,
        k: int,
        m: int,
        ku: int,
        kl: int,
        block_size_inner: int
):
    
    pos = iter*block_size_inner
    band_limit_upper = max(0, pos - ku)
    band_limit_lower = min(pos + kl + block_size_inner, m)
    
    D1 = slice(pos, min(k, pos+block_size_inner))

    if pos < (ku+1):
        _A1 = slice(band_limit_upper, band_limit_upper)
        A1 = None
    else:
        _A1 = slice(band_limit_upper, band_limit_upper+block_size_inner)
        A1 = _A1

    if pos > (k-kl-1):
        _A3 = slice(band_limit_lower, band_limit_lower)
        A3 = None
    else:
        _A3 = slice(band_limit_lower-block_size_inner, band_limit_lower)
        A3 = _A3

    A2 = slice(_A1.stop, _A3.start)

    return D1, A1, A2, A3

def _gbmm_gpu_inner(
        E: np.ndarray,
        A: np.ndarray,
        D: np.ndarray,
        ku: int,
        kl: int,
        block_size_inner,
        streams,
        events
    ) -> np.ndarray:

    k = D.shape[0]
    n = D.shape[1]
    m = A.shape[0]

    #print("#########################################################")
    #print("PARAMETERS -----------------")
    #print("Shapes: E = ", E.shape, " A = ", A.shape, " D = ", D.shape)
    #print("block_size_outer = ", block_size_outer, " block_size_inner = ", block_size_inner)
    #print("A -> ku = ", ku, " A -> kl = ", kl )
    #print("----------------------------")

    # Number of blocks to compute
    num_blocks = -(-k//block_size_inner) #Rounded up

    # Streams
    # streams = [cp.cuda.Stream(non_blocking=True) for _ in range(2)]
    # Events
    # events = [cp.cuda.Event() for _ in range(2)]

    # Buffer
    D1 = [cp.empty((block_size_inner, n)) for _ in range(2)]

    A11 = [cp.empty((block_size_inner, block_size_inner)) for _ in range(2)]
    A21 = [cp.empty((ku+kl+block_size_inner, block_size_inner)) for _ in range(2)]
    A31 = [cp.empty((block_size_inner, block_size_inner)) for _ in range(2)]

    #E1 = [cp.empty((block_size_inner, block_size_outer)) for _ in range(2)]
    #E2 = [cp.empty((ku+kl+block_size_inner, block_size_outer)) for _ in range(2)]
    #E3 = [cp.empty((block_size_inner, block_size_outer)) for _ in range(2)]

    # Iteration step i = 0

    #print("SLICES ---------------------")
    #for i in range(0,num_blocks):
    #    D1_x1, A1_x1, A2_x1, A3_x1 = _slicer(i, k, m, ku, kl, block_size_inner)
    #    print("D1 = ", D1_x1, " A1 = ", A1_x1, " A2 = ", A2_x1, " A3 = ", A3_x1)
    #print("----------------------------")
    
    with streams[0] as stream:

        D1_x1, A1_x1, A2_x1, A3_x1 = _slicer(0, k, m, ku, kl, block_size_inner)

        #print("LOOP ", 0, "----------------")
        #print("D1 = ", D1_x1, " A1 = ", A1_x1, " A2 = ", A2_x1, " A3 = ", A3_x1)
        #print("----------------------------")

        D1[0][:D1_x1.stop-D1_x1.start] = cp.asarray(D[D1_x1, :])
        D1_cur = D1[0][:D1_x1.stop-D1_x1.start]

        A21[0][:A2_x1.stop-A2_x1.start,:D1_x1.stop-D1_x1.start] = cp.asarray(A[A2_x1,D1_x1])
        A21_cur = A21[0][:A2_x1.stop-A2_x1.start,:D1_x1.stop-D1_x1.start]
        
        if A1_x1 is not None and A3_x1 is not None:
            A11[0][:A1_x1.stop-A1_x1.start,:D1_x1.stop-D1_x1.start] = cp.asarray(A[A1_x1,D1_x1])
            A11_cur = A11[0][:A1_x1.stop-A1_x1.start,:D1_x1.stop-D1_x1.start]

            A31[0][:A3_x1.stop-A3_x1.start,:D1_x1.stop-D1_x1.start] = cp.asarray(A[A3_x1,D1_x1])
            A31_cur = A31[0][:A3_x1.stop-A3_x1.start,:D1_x1.stop-D1_x1.start]

            E[A1_x1, :] = E[A1_x1, :] + cp.asnumpy(A11_cur @ D1_cur)
            E[A2_x1, :] = E[A2_x1, :] + cp.asnumpy(A21_cur @ D1_cur)
            E[A3_x1, :] = E[A3_x1, :] + cp.asnumpy(A31_cur @ D1_cur)
            #print("1-Real:", 0, "\n", E)
            events[0].record(stream=stream)
 

        elif A1_x1 is not None and A3_x1 is None: 
            A11[0][:A1_x1.stop-A1_x1.start,:D1_x1.stop-D1_x1.start] = cp.asarray(A[A1_x1,D1_x1])
            A11_cur = A11[0][:A1_x1.stop-A1_x1.start,:D1_x1.stop-D1_x1.start]

            E[A1_x1, :] = E[A1_x1, :] + cp.asnumpy(A11_cur @ D1_cur)
            E[A2_x1, :] = E[A2_x1, :] + cp.asnumpy(A21_cur @ D1_cur)
            #print("2-Real:", 0, "\n", E)
            events[0].record(stream=stream)

        elif A1_x1 is None and A3_x1 is not None:
            A31[0][:A3_x1.stop-A3_x1.start,:D1_x1.stop-D1_x1.start] = cp.asarray(A[A3_x1,D1_x1])
            A31_cur = A31[0][:A3_x1.stop-A3_x1.start,:D1_x1.stop-D1_x1.start]

            E[A2_x1, :] = E[A2_x1, :] + cp.asnumpy(A21_cur @ D1_cur)
            E[A3_x1, :] = E[A3_x1, :] + cp.asnumpy(A31_cur @ D1_cur)
            #print("3-Real:", 0, "\n", E)
            events[0].record(stream=stream)

        else:
            E[A2_x1, :] = E[A2_x1, :] + cp.asnumpy(A21_cur @ D1_cur)
            #print("4-Real:", 0, "\n", E)
            events[0].record(stream=stream)

    

    # Iteration step i = 1 to i = num_blocks-1
    for i in range(1, num_blocks):

        D1_x1, A1_x1, A2_x1, A3_x1 = _slicer(i, k, m, ku, kl, block_size_inner)

        #print("SLICES ---------------------")
        #print("Loop Nr: ", i)
        #print("D1 = ", D1_x1, " A1 = ", A1_x1, " A2 = ", A2_x1, " A3 = ", A3_x1)
        

        with streams[i % 2] as stream:

            #print("Stream Nr. ", (i % 2), ": ", stream)
            #print("----------------------------")
            
            D1[i % 2][:D1_x1.stop-D1_x1.start] = cp.asarray(D[D1_x1, :])
            D1_cur = D1[i % 2][:D1_x1.stop-D1_x1.start]

            A21[i % 2][:A2_x1.stop-A2_x1.start,:D1_x1.stop-D1_x1.start] = cp.asarray(A[A2_x1,D1_x1])
            A21_cur = A21[i % 2][:A2_x1.stop-A2_x1.start,:D1_x1.stop-D1_x1.start]
            
            if A1_x1 is not None and A3_x1 is not None:
                A11[i % 2][:A1_x1.stop-A1_x1.start,:D1_x1.stop-D1_x1.start] = cp.asarray(A[A1_x1,D1_x1])
                A11_cur = A11[i % 2][:A1_x1.stop-A1_x1.start,:D1_x1.stop-D1_x1.start]

                A31[i % 2][:A3_x1.stop-A3_x1.start,:D1_x1.stop-D1_x1.start] = cp.asarray(A[A3_x1,D1_x1])
                A31_cur = A31[i % 2][:A3_x1.stop-A3_x1.start,:D1_x1.stop-D1_x1.start]

                stream.wait_event(event=events[(i-1) % 2])

                E[A1_x1, :] = E[A1_x1, :] + cp.asnumpy(A11_cur @ D1_cur)
                E[A2_x1, :] = E[A2_x1, :] + cp.asnumpy(A21_cur @ D1_cur)
                E[A3_x1, :] = E[A3_x1, :] + cp.asnumpy(A31_cur @ D1_cur)
                #print("1-Real:", i, "\n", E)

                events[i % 2].record(stream=stream)

            elif A1_x1 is not None and A3_x1 is None: 
                A11[i % 2][:A1_x1.stop-A1_x1.start,:D1_x1.stop-D1_x1.start] = cp.asarray(A[A1_x1,D1_x1])
                A11_cur = A11[i % 2][:A1_x1.stop-A1_x1.start,:D1_x1.stop-D1_x1.start]

                stream.wait_event(event=events[(i-1) % 2])

                E[A1_x1, :] = E[A1_x1, :] + cp.asnumpy(A11_cur @ D1_cur)
                E[A2_x1, :] = E[A2_x1, :] + cp.asnumpy(A21_cur @ D1_cur)
                #print("2-Real:", i, "\n", E)

                events[i % 2].record(stream=stream)


            elif A1_x1 is None and A3_x1 is not None:
                A31[i % 2][:A3_x1.stop-A3_x1.start,:D1_x1.stop-D1_x1.start] = cp.asarray(A[A3_x1,D1_x1])
                A31_cur = A31[i % 2][:A3_x1.stop-A3_x1.start,:D1_x1.stop-D1_x1.start]

                stream.wait_event(event=events[(i-1) % 2])

                E[A2_x1, :] = E[A2_x1, :] + cp.asnumpy(A21_cur @ D1_cur)
                E[A3_x1, :] = E[A3_x1, :] + cp.asnumpy(A31_cur @ D1_cur)
                #print("3-Real:", i, "\n", E)

                events[i % 2].record(stream=stream)

            else:
                stream.wait_event(event=events[(i-1) % 2])

                E[A2_x1, :] = E[A2_x1, :] + cp.asnumpy(A21_cur @ D1_cur)
                #print("4-Real:", i, "\n", E)

                events[i % 2].record(stream=stream)
    
    #print("Ref:\n", A @ D)
    return E

def  gbmm_gpu(
        A: np.ndarray,
        ku_A: int,
        kl_A: int,
        B: np.ndarray,
        ku_B: int,
        kl_B: int,
        block_size_outer,
        block_size_inner
    ):
    C = np.zeros((A.shape[0], B.shape[1]))
    C = _gbmm_gpu_outer(C, A, ku_A, kl_A, B, ku_B, kl_B, block_size_outer, block_size_inner)
    return C

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
        A = banded_matrix_generator(10000, 2400, 2900)
        B = banded_matrix_generator(10000, 3000, 800)

        print("Calculating gbmm_gpu")
        C = gbmm_gpu(A, 2400, 2900, B, 3000, 8000, 300, 200)
    else:
        print("Debug Setup Used")
        print("Generating band matrices")
        A = banded_matrix_generator(10, 2, 2)
        B = banded_matrix_generator(10, 0, 2)

        print("Calculating gbmm_gpu")
        C = gbmm_gpu(A, 2, 2, B, 0, 2, 3, 2)

    print("Calculating Ref with numpy")
    T = A @ B
    print(C-T)
    assert np.allclose(C, T)
    print("Correct Result computed")