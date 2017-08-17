import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import numpy as np
from skcuda import linalg, rlinalg
linalg.init()
rlinalg.init()
#Randomized SVD decomposition of the square matrix `a` with single precision.
#Note: There is no gain to use rsvd if k > int(n/1.5)
a = np.array(np.random.randn(5, 5), np.float32, order='F')
a_gpu = gpuarray.to_gpu(a)
U, s, Vt = rlinalg.rsvd(a_gpu, k=5, method='standard')
np.allclose(a, np.dot(U.get(), np.dot(np.diag(s.get()), Vt.get())), 1e-4)