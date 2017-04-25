import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
from skcuda.fft import fft, ifft, Plan

#wrap for fft in cufft
def fftnc2c_cuda( x ):
    x = np.array(x).astype(np.complex64)
    x_gpu  = gpuarray.to_gpu(x)
    xf_gpu = gpuarray.empty(x.shape, np.complex64)
    plan   = Plan(x.shape, np.complex64, np.complex64)
    fft(x_gpu, xf_gpu, plan)
    return xf_gpu.get()

#wrap for ifft in cufft, noted that the cufft functions are not normalized
#here I normalize ifft by 1/N
def ifftnc2c_cuda( x ):
    x = np.array(x).astype(np.complex64)
    x_gpu  = gpuarray.to_gpu(x)
    xf_gpu = gpuarray.empty(x.shape, np.complex64)
    plan   = Plan(x.shape, np.complex64, np.complex64)
    ifft(x_gpu, xf_gpu, plan)
    return xf_gpu.get()/np.prod(x.shape)

#test 1d fft
def test1():
    N = 128
    x = np.asarray(np.random.rand(N), np.complex64)
    xf = np.fft.fft(x)
    x_gpu = gpuarray.to_gpu(x)
    xf_gpu = gpuarray.empty(N, np.complex64)
    plan = Plan(x.shape, np.complex64, np.complex64)
    fft(x_gpu, xf_gpu, plan)
    print(np.allclose(xf[0:N], xf_gpu.get(), atol=1e-3))

#test 2d fft2
def test2():
    N = 128
    x = np.asarray(np.random.rand(N,N), np.complex64)
    xf = np.fft.fft2(x)
    x_gpu = gpuarray.to_gpu(x)
    xf_gpu = gpuarray.empty((N,N), np.complex64)
    plan = Plan(x.shape, np.complex64, np.complex64)
    fft(x_gpu, xf_gpu, plan)
    print(np.allclose(xf[0:N, 0:N], xf_gpu.get(), atol=1e-3))

#test 3d fft3
def test3():
    N = 128
    x = np.asarray(np.random.rand(N,N,N), np.complex64)
    xf = np.fft.fftn(x, s=None, axes=(0,1,2))
    x_gpu = gpuarray.to_gpu(x)
    xf_gpu = gpuarray.empty((N,N,N), np.complex64)
    plan = Plan(x.shape, np.complex64, np.complex64)
    fft(x_gpu, xf_gpu, plan)
    print(np.allclose(xf[0:N, 0:N,0:N], xf_gpu.get(), atol=1e-2))

def test4():
    N = 128
    x = np.asarray(np.random.rand(N,N,32), np.complex64)
    xf = np.fft.fftn(x, s=None, axes=(0,1,2))    
    print(np.allclose(xf, fftnc2c_cuda(x), atol=1e-2))