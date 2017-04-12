from numba import cuda
import numba
import numpy as np
bpg = 10
tpb = 10

n = bpg * tpb

@cuda.jit
def cu_square_matrix_mul(A, B, C):
    sA = cuda.shared.array(shape=(tpb, tpb), dtype=numba.float32)
    sB = cuda.shared.array(shape=(tpb, tpb), dtype=numba.float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y

    x = tx + bx * bw
    y = ty + by * bh


    acc = np.float32(0.) #cuda.local.array(shape=(1), dtype=np.float32) #0.
    for i in range(bpg):
        if x < n and y < n:
            sA[ty, tx] = A[y, tx + i * tpb]
            sB[ty, tx] = B[ty + i * tpb, x]

        cuda.syncthreads()

        if x < n and y < n:
            for j in range(tpb):
                acc += sA[ty, j] * sB[j, tx]

        cuda.syncthreads()

    if x < n and y < n:
        C[y, x] = acc

if __name__ == "__main__":

    print 'Blocks per grid:', bpg
    print 'Threads per block', tpb

    A = np.zeros((10,10),dtype=np.float32)
    B = A
    C = A
    cu_square_matrix_mul[bpg, tpb](A, B, C)
    print C
