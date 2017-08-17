import numpy as np
import scipy.signal as ss
"""
def array_outer_product(A, B, result=None):
    ''' Compute the outer-product in the final two dimensions of the given arrays.
    If the result array is provided, the results are written into it.
    '''
    assert(A.shape[:-1] == B.shape[:-1])
    if result is None:
        result=np.zeros(A.shape+B.shape[-1:], dtype=A.dtype)
    if A.ndim==1:
        result[:,:]=np.outer(A, B)
    else:
        for idx in range(A.shape[0]):
            array_outer_product(A[idx,...], B[idx,...], result[idx,...])
    return result
"""
def hanning2d( a, b ):
    # build 2d window
    w2d = np.outer(ss.hanning(a), ss.hanning(b))
    return np.sqrt(w2d)

def hanning3d( a, b, c ):
    w2d = np.outer(ss.hanning(a), ss.hanning(b))
    w3d = np.zeros((a,b,c))
    for i in range(a):
        w3d[i, :, :] = np.outer(w2d[i,:].flatten(), ss.hanning(c))
    return w3d**(1.0/3.0)

def hamming2d( a, b ):
    # build 2d window
    w2d = np.outer(ss.hamming(a), ss.hamming(b))
    return np.sqrt(w2d)

def hamming3d( a, b, c ):
    w2d = np.outer(ss.hamming(a), ss.hamming(b))
    w3d = np.zeros((a,b,c))
    for i in range(a):
        w3d[i, :, :] = np.outer(w2d[i,:].flatten(), ss.hamming(c))
    return w3d**(1.0/3.0)