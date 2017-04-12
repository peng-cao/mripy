from __future__ import division
import numpy as np
from numbapr0 import *


# Device Functions
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Works and can be called corrently from TestKernel Scalar
@cuda.jit('float32(float32, float32)', device=True)
def myfuncScalar(a, b):
    return a+b;


# Works and can be called correctly from TestKernel Array
@cuda.jit('float32[:](float32[:])', device=True)
def myfuncArray(A):
    for k in xrange(4):
        A[k] += 2*k;
    return A


# Takes Matrix A and Vector v, multiplies them and returns a vector of shape v. Does not even compile.
# Failed at nopython (nopython frontend), Only accept returning of array passed into the function as argument
# But v is passed to the function as argument...

@cuda.jit('float32[:](float32[:,:], float32[:])', device=True)
def MatrixMultiVector(A,v):
    tmp = cuda.local.array(shape=4, dtype=float32); # is that thing even empty? It could technically be anything, right?
    for i in xrange(A[0].size):
        for j in xrange(A[1].size):
            tmp[i] += A[i][j]*v[j];
    v = tmp;
    return v;



# Kernels
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# TestKernel Scalar - Works
@cuda.jit(void(float32[:,:]))
def TestKernelScalar(InputArray):
    i = cuda.grid(1)
    for j in xrange(InputArray[1].size):
        InputArray[i,j] = myfuncScalar(5,7);


# TestKernel Array
@cuda.jit(void(float32[:,:]))
def TestKernelArray(InputArray):

    # Defining arrays this way seems super tedious, there has to be a better way.
    M = cuda.local.array(shape=4, dtype=np.float32);
    M[0] = 1; M[1] = 0; M[2] = 0; M[3] = 0;

    tmp = myfuncArray(M);
    #tmp = MatrixMultiVector(A,M); -> we still have to define a 4x4 matrix for that.

    i = cuda.grid(1)
    for j in xrange(InputArray[1].size):
        InputArray[i,j] += tmp[j];

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Main
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

N = 4;

C = np.zeros((N,N), dtype=float32);
TestKernelArray[1,N](C);

print(C)