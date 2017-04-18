import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
from skcuda.fft import fft, ifft, Plan

def fftnc2c_cuda( x ):
    x = np.array(x).astype(np.complex64)
    x_gpu  = gpuarray.to_gpu(x)
    xf_gpu = gpuarray.empty(x.shape, np.complex64)
    plan   = Plan(x.shape, np.complex64, np.complex64)
    fft(x_gpu, xf_gpu, plan)  
    return xf_gpu.get()


def ifftnc2c_cuda( x ):
    x = np.array(x).astype(np.complex64)
    x_gpu  = gpuarray.to_gpu(x)
    xf_gpu = gpuarray.empty(x.shape, np.complex64)
    plan   = Plan(x.shape, np.complex64, np.complex64)
    ifft(x_gpu, xf_gpu, plan)  
    return xf_gpu.get()
     