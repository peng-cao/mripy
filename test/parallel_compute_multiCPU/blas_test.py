import numpy as np
import scipy.linalg.blas as blas
import utilities.utilities_class as utc
def test():
    N = 512
    a = np.ones((N,N,N))
    b = 2*1j*np.ones((N,N,N))
    time = utc.timing()
    time.start()
    for _ in range(10):
         #b = blas.caxpy(a, b)
         b = a + b
    time.stop().display()
    #print(b)