from numba import cuda
import numpy as np
@cuda.jit('void(float32[:], float32[:], float32[:])')
def cu_add2(a, b, c):
    """This kernel function will be executed by a thread."""
    i  = cuda.grid(1)

    if i > a.shape[0]:
        return
    #d = [[1, 1, 0.],[1, 1, 0.],[0., 0., 1.]]
    #e = [0., 0., 1.]
    #c = cuda.local.array(shape=(100), dtype=np.float32)
    c[i] = a[i] + b[i]
    c[i+10] = a[i] + b[i]

device = cuda.get_current_device()

n = 10
a = np.arange(n, dtype=np.float32)
b = np.arange(n, dtype=np.float32)
c = np.zeros(n+10, dtype=np.float32)#empty_like(a)

tpb = device.WARP_SIZE
bpg = int(np.ceil(float(n)/tpb))
print 'Blocks per grid:', bpg
print 'Threads per block', tpb

cu_add2[bpg, tpb](a, b, c)
print c