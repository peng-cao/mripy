from numba import vectorize, float32,cuda
import numpy as np
import time
import os
from math import cos, sin

@vectorize([float32(float32, float32)], target='cuda')
def g(x, y):
    phi = np.pi

    return x + y

def main():  
    N = 320
    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)

    t0 = time.clock()

    C = g(A, B)

    t1 = time.clock()
    delta_t = t1 - t0

    print('g executed in {0} seconds'.format(delta_t))

if __name__ == '__main__':
    main()
    #numba.cuda.profile_stop()