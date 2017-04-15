from __future__ import print_function, division
import numpy as np
import numba
from numba import cuda
from time import time
from math import exp
import utilities.utilities_func as ut
import nufft_func as unfft_func
#import matplotlib.pyplot as plt


# this implementation is very slow due to the for loop used in build_grid_1d1_cuda_1, 
# should be accelerated by jit and percomputing of some exponentials
# kernal for griding in cuda
@cuda.jit
def gaussker_1d1_cuda_1(x, m, c, hx, nf1, nspread, tau, adftau, mm ):
    """This kernel function for gauss grid 1d typ1, and it will be executed by a thread."""
    i  = cuda.grid(1)
    if i > x.shape[0]:
        return
    #compute griding here
    adftau[i] = c[i] * exp(-0.25 * (x[i] % (2 * np.pi) - hx * (m[i] + mm)) ** 2 / tau)

# do griding with cuda acceleration
#@numba.jit(nopython=True)
def build_grid_1d1_cuda_1( x, c, tau, nspread, ftau ):
    nf1 = ftau.shape[0]
    hx = 2 * np.pi / nf1    
    device = cuda.get_current_device()
    n = x.shape[0] #number of kernels in the computing
    tpb = device.WARP_SIZE
    bpg = int(np.ceil(float(n)/tpb))
    #print ('Blocks per grid: %d' % bpg)
    #print ('Threads per block: %d'% tpb)
    #due to the memory conflict, I moved the sumation out of the cuda kernel 
    adftau = np.zeros(c.shape,dtype=ftau.dtype)#get the kernel results from cuda parallel computing   
    m = np.zeros(x.shape,dtype=np.int)#the index of the closest grid for x
    for i in range(n): #compute the index m
        xi = x[i] % (2 * np.pi) #x, shift the source point xj so that it lies in [0,2*pi]    
        m[i] = 1 + int(xi // hx) #index for the closest grid point  
    for mm in range(-nspread, nspread):
        #computing the kernel on gpu
        gaussker_1d1_cuda_1[bpg, tpb](x, m, c, hx, nf1, nspread, tau, adftau, mm ) 
        #do accumulative sum on cpu, due to the memory conflict, when doing it on gpu
        for i in range(n):
            ftau[(m[i]+mm) % nf1] += adftau[i]
    return ftau


# this function has memory conflict in gpu, due to the overlapped ftau when doing griding
# kernal for griding in cuda
#@cuda.jit('void(float64[:], complex128[:], float64, int32, int32, float64, complex128[:])')
@cuda.jit
def gaussker_1d1_cuda(x, c, hx, nf1, nspread, tau, ftau ):
    """This kernel function for gauss grid 1d typ1, and it will be executed by a thread."""
    i  = cuda.grid(1)
    if i > x.shape[0]:
        return
    #do the 1d griding here
    xi = x[i] % (2 * np.pi) #x, shift the source point xj so that it lies in [0,2*pi] 
    m1 = 1 + int(xi // hx) #index for the closest grid point
    for mm1 in range(-nspread, nspread): #mm index for all the spreading points
        #griding with g(x) = exp(-(x^2) / 4*tau)
        ftau[(m1 + mm1) % nf1] += c[i] * exp(-0.25 * ((xi - hx * (m1 + mm1)) ** 2 ) / tau) 

    
# do griding with cuda acceleration
def build_grid_1d1_cuda( x, c, tau, nspread, ftau ):
    nf1 = ftau.shape[0]
    hx = 2 * np.pi / nf1
    device = cuda.get_current_device()
    n = x.shape[0] #number of kernels in the computing
    tpb = device.WARP_SIZE
    bpg = int(np.ceil(float(n)/tpb))
    gaussker_1d1_cuda[bpg, tpb](x, c, hx, nf1, nspread, tau, ftau)
    return ftau

# this function has memory conflict in gpu, due to the overlapped ftau when doing griding
# kernal for griding in cuda
#@cuda.jit('void(float64[:], complex128[:], float64, int32, int32, float64, complex128[:])')
@cuda.jit
def gaussker_1d1_cuda1(x, c, hx, nf1, nspread, tau, ftau, mm1 ):
    """This kernel function for gauss grid 1d typ1, and it will be executed by a thread."""
    #sftau = cuda.shared.array(shape=320, dtype=numba.complex128)
    i  = cuda.grid(1)
    if i > x.shape[0]:
        return
    #do the 1d griding here
    xi = x[i] % (2 * np.pi) #x, shift the source point xj so that it lies in [0,2*pi] 
    m1 = 1 + int(xi // hx) #index for the closest grid point
    #for mm1 in range(-nspread, nspread): #mm index for all the spreading point

    ftau[(m1 + mm1) % nf1] += c[i] * exp(-0.25 * ((xi - hx * (m1 + mm1)) ** 2 ) / tau) 

    
# do griding with cuda acceleration
def build_grid_1d1_cuda1( x, c, tau, nspread, ftau ):
    nf1 = ftau.shape[0]
    hx = 2 * np.pi / nf1
    device = cuda.get_current_device()
    n = x.shape[0] #number of kernels in the computing
    tpb = device.WARP_SIZE
    bpg = int(np.ceil(float(n)/tpb))

    for mm1 in range(-nspread, nspread):             
        gaussker_1d1_cuda1[bpg, tpb](x, c, hx, nf1, nspread, tau, ftau, mm1 )
    return ftau

# this function has memory conflict in gpu, due to the overlapped ftau when doing griding
# kernal for griding in cuda
#@cuda.jit('void(float64[:], complex128[:], float64, int32, int32, float64, complex128[:])')
@cuda.jit
def gaussker_1d1_cuda2(x, c, hx, nf1, nspread, tau, ftau, tmp, idx ):
    i  = cuda.grid(1)
    #tx = cuda.threadIdx.x
    if i > x.shape[0]:
        return
    #do the 1d griding here
    xi = x[i] % (2 * np.pi) #x, shift the source point xj so that it lies in [0,2*pi] 
    m1 = 1 + int(xi // hx) #index for the closest grid point
    for mm1 in range(-nspread, nspread): #mm index for all the spreading points
        #griding with g(x) = exp(-(x^2) / 4*tau)
        tmp[i,mm1] = c[i] * exp(-0.25 * ((xi - hx * (m1 + mm1)) ** 2 ) / tau)
        idx[i,mm1] = (m1 + mm1) % nf1 
    #cuda.syncthreads()
    #for mm1 in range(-nspread, nspread): #mm index for all the spreading points                     
    #    ftau[(m1 + mm1) % nf1] += tmp[i,mm1] 
    
# do griding with cuda acceleration
def build_grid_1d1_cuda2( x, c, tau, nspread, ftau ):
    nf1 = ftau.shape[0]
    hx = 2 * np.pi / nf1
    device = cuda.get_current_device()
    n = x.shape[0] #number of kernels in the computing
    tpb = device.WARP_SIZE
    bpg = int(np.ceil(float(n)/tpb))
    tmp = np.zeros((n,2*nspread+1),dtype=c.dtype)
    idx =  np.zeros((n,2*nspread+1),dtype=np.int)
    gaussker_1d1_cuda2[bpg, tpb](x, c, hx, nf1, nspread, tau, ftau, tmp, idx )
    for x in range(n):
        for mm1 in range(-nspread, nspread):
            ftau[idx[x,mm1]] += tmp[x,mm1] 
    return ftau

@cuda.jit
def gaussker_1d1_cuda3(x, c, hx, nf1, nspread, tau, ftau ):
    """This kernel function for gauss grid 1d typ1, and it will be executed by a thread."""
    #sftau = cuda.shared.array(shape=320, dtype=numba.complex128)
    q  = cuda.grid(1)
    if q > nf1:
        return
    #do the 1d griding here
    for i in range(x.shape[0]):
        xi = x[i] % (2 * np.pi) #x, shift the source point xj so that it lies in [0,2*pi] 
        m1 = 1 + int(xi // hx) #index for the closest grid point
        for mm1 in range(-nspread, nspread): #mm index for all the spreading point
            if (((m1 + mm1) % nf1)== q):
                ftau[q] += c[i] * exp(-0.25 * ((xi - hx * (m1 + mm1)) ** 2 ) / tau) 

    
# do griding with cuda acceleration
def build_grid_1d1_cuda3( x, c, tau, nspread, ftau ):
    nf1 = ftau.shape[0]
    hx = 2 * np.pi / nf1
    device = cuda.get_current_device()
    n = nf1#x.shape[0] #number of kernels in the computing
    tpb = device.WARP_SIZE
    bpg = int(np.ceil(float(n)/tpb))
             
    gaussker_1d1_cuda3[bpg, tpb](x, c, hx, nf1, nspread, tau, ftau )
    return ftau


# type 2
# kernal for griding in cuda
@cuda.jit
def gaussker_1d2_cuda(x, c, hx, nf1, nspread, tau, fntau ):
    """This kernel function for gauss grid 1d typ1, and it will be executed by a thread."""
    i  = cuda.grid(1)
    if i > x.shape[0]:
        return
    #do the 1d griding here
    xi = x[i] % (2 * np.pi) #x, shift the source point xj so that it lies in [0,2*pi] 
    m1 = 1 + int(xi // hx) #index for the closest grid point
    for mm1 in range(-nspread, nspread): #mm index for all the spreading points
        #griding with g(x,y) = exp(-(x^2 + y^2) / 4*tau)
        c[i] += fntau[(m1 + mm1) % nf1] * exp(-0.25 * ((xi - hx * (m1 + mm1)) ** 2 ) / tau) 
    
# do griding with cuda acceleration
#@numba.jit(nopython=True)
def build_grid_1d2_cuda( x, fntau, tau, nspread ):
    nf1 = fntau.shape[0]
    hx = 2 * np.pi / nf1
    c = np.zeros(x.shape,dtype = fntau.dtype)
    device = cuda.get_current_device()
    n = x.shape[0] #number of kernels in the computing
    tpb = device.WARP_SIZE
    bpg = int(np.ceil(float(n)/tpb))
    gaussker_1d2_cuda[bpg, tpb](x, c, hx, nf1, nspread, tau, fntau )
    return c/nf1

# type 1
# kernal for griding in cuda
#@cuda.jit('void(float64[:], complex128[:], float64, int32, int32, float64, complex128[:])')
@cuda.jit
def gaussker_2d1_cuda(x, y, c, hx, hy, nf1, nf2, nspread, tau, ftau ):
    """This kernel function for gauss grid 1d typ1, and it will be executed by a thread."""
    i  = cuda.grid(1)
    if i > x.shape[0]:
        return
    #do the 1d griding here
    xi = x[i] % (2 * np.pi) #x, shift the source point xj so that it lies in [0,2*pi] 
    yi = y[i] % (2 * np.pi) #y, shift the source point yj so that it lies in [0,2*pi] 
    m1 = 1 + int(xi // hx) #index for the closest grid point
    m2 = 1 + int(yi // hy) #index for the closest grid point
    for mm1 in range(-nspread, nspread): #mm index for all the spreading points
        for mm2 in range(-nspread,nspread):
            #griding with g(x,y) = exp(-(x^2 + y^2) / 4*tau)
            ftau[(m1 + mm1) % nf1, (m2 + mm2) % nf2] \
            += c[i] * exp(-0.25 * (\
            (xi - hx * (m1 + mm1)) ** 2 + \
            (yi - hy * (m2 + mm2)) ** 2 ) / tau) 
    
# do griding with cuda acceleration
def build_grid_2d1_cuda( x, y, c, tau, nspread, ftau ):
    nf1 = ftau.shape[0]
    nf2 = ftau.shape[1]
    hx = 2 * np.pi / nf1
    hy = 2 * np.pi / nf2 
    device = cuda.get_current_device()
    n = x.shape[0] #number of kernels in the computing
    tpb = device.WARP_SIZE
    bpg = int(np.ceil(float(n)/tpb))
    gaussker_2d1_cuda[bpg, tpb](x, y, c, hx, hy, nf1, nf2, nspread, tau, ftau)
    return ftau

# type 2
# kernal for griding in cuda
@cuda.jit
def gaussker_2d2_cuda(x, y, c, hx, hy, nf1, nf2, nspread, tau, fntau ):
    """This kernel function for gauss grid 1d typ1, and it will be executed by a thread."""
    i  = cuda.grid(1)
    if i > x.shape[0]:
        return
    #do the 1d griding here
    xi = x[i] % (2 * np.pi) #x, shift the source point xj so that it lies in [0,2*pi] 
    yi = y[i] % (2 * np.pi) #y, shift the source point yj so that it lies in [0,2*pi] 
    m1 = 1 + int(xi // hx) #index for the closest grid point
    m2 = 1 + int(yi // hy) #index for the closest grid point
    for mm1 in range(-nspread, nspread): #mm index for all the spreading points
        for mm2 in range(-nspread,nspread):
            #griding with g(x,y) = exp(-(x^2 + y^2) / 4*tau)
            c[i] \
            += fntau[(m1 + mm1) % nf1, (m2 + mm2) % nf2] * exp(-0.25 * (\
            (xi - hx * (m1 + mm1)) ** 2 + \
            (yi - hy * (m2 + mm2)) ** 2 ) / tau) 
    
# do griding with cuda acceleration
#@numba.jit(nopython=True)
def build_grid_2d2_cuda( x, y, fntau, tau, nspread ):
    nf1 = fntau.shape[0]
    nf2 = fntau.shape[1]
    hx = 2 * np.pi / nf1
    hy = 2 * np.pi / nf2 
    c = np.zeros(x.shape,dtype = fntau.dtype)
    device = cuda.get_current_device()
    n = x.shape[0] #number of kernels in the computing
    tpb = device.WARP_SIZE
    bpg = int(np.ceil(float(n)/tpb))
    gaussker_2d2_cuda[bpg, tpb](x, y, c, hx, hy, nf1, nf2, nspread, tau, fntau )
    return c/(nf1*nf2)


# kernal for griding in cuda
#@cuda.jit('void(float64[:], complex128[:], float64, int32, int32, float64, complex128[:])')
@cuda.jit
def gaussker_3d1_cuda(x, y, z, c, hx, hy, hz, nf1, nf2, nf3, nspread, tau, ftau ):
    """This kernel function for gauss grid 1d typ1, and it will be executed by a thread."""
    i  = cuda.grid(1)
    if i > x.shape[0]:
        return
    #do the 1d griding here
    xi = x[i] % (2.0 * np.pi) #x, shift the source point xj so that it lies in [0,2*pi] 
    yi = y[i] % (2.0 * np.pi) #y, shift the source point yj so that it lies in [0,2*pi]
    zi = z[i] % (2.0 * np.pi) #z, shift the source point zj so that it lies in [0,2*pi] 
    m1 = 1 + int(xi // hx) #index for the closest grid point
    m2 = 1 + int(yi // hy) #index for the closest grid point
    m3 = 1 + int(zi // hz) #index for the closest grid point
    for mm1 in range(-nspread, nspread): #mm index for all the spreading points
        for mm2 in range(-nspread,nspread):
            for mm3 in range(-nspread,nspread):
                #griding with g(x,y) = exp(-(x^2 + y^2) / 4*tau)
                ftau[(m1 + mm1) % nf1, (m2 + mm2) % nf2, (m3 + mm3) % nf3] \
                += c[i] * exp(-0.25 * (\
                (xi - hx * (m1 + mm1)) ** 2 + \
                (yi - hy * (m2 + mm2)) ** 2 + \
                (zi - hz * (m3 + mm3)) ** 2 ) / tau)
    
# do griding with cuda acceleration
def build_grid_3d1_cuda( x, y, z, c, tau, nspread, ftau ):
    nf1 = ftau.shape[0]
    nf2 = ftau.shape[1]
    nf3 = ftau.shape[2]
    hx = 2 * np.pi / nf1
    hy = 2 * np.pi / nf2 
    hz = 2 * np.pi / nf3
    device = cuda.get_current_device()
    n = x.shape[0] #number of kernels in the computing
    tpb = device.WARP_SIZE
    bpg = int(np.ceil(float(n)/tpb))
    gaussker_3d1_cuda[bpg, tpb](x, y, z, c, hx, hy, hz, nf1, nf2, nf3, nspread, tau, ftau)
    return ftau


# kernal for griding in cuda
#@cuda.jit('void(float64[:], complex128[:], float64, int32, int32, float64, complex128[:])')
@cuda.jit
def gaussker_3d2_cuda(x, y, z, c, hx, hy, hz, nf1, nf2, nf3, nspread, tau, fntau ):
    """This kernel function for gauss grid 1d typ1, and it will be executed by a thread."""
    i  = cuda.grid(1)
    if i > x.shape[0]:
        return
    #do the 1d griding here
    xi = x[i] % (2.0 * np.pi) #x, shift the source point xj so that it lies in [0,2*pi] 
    yi = y[i] % (2.0 * np.pi) #y, shift the source point yj so that it lies in [0,2*pi]
    zi = z[i] % (2.0 * np.pi) #z, shift the source point zj so that it lies in [0,2*pi] 
    m1 = 1 + int(xi // hx) #index for the closest grid point
    m2 = 1 + int(yi // hy) #index for the closest grid point
    m3 = 1 + int(zi // hz) #index for the closest grid point
    for mm1 in range(-nspread, nspread): #mm index for all the spreading points
        for mm2 in range(-nspread,nspread):
            for mm3 in range(-nspread,nspread):
                #griding with g(x,y) = exp(-(x^2 + y^2) / 4*tau)
                c[i] \
                += fntau[(m1 + mm1) % nf1, (m2 + mm2) % nf2, (m3 + mm3) % nf3] * exp(-0.25 * (\
                (xi - hx * (m1 + mm1)) ** 2 + \
                (yi - hy * (m2 + mm2)) ** 2 + \
                (zi - hz * (m3 + mm3)) ** 2 ) / tau)

# do griding with cuda acceleration
def build_grid_3d2_cuda( x, y, z, fntau, tau, nspread ):
    nf1 = fntau.shape[0]
    nf2 = fntau.shape[1]
    nf3 = fntau.shape[2]
    hx = 2 * np.pi / nf1
    hy = 2 * np.pi / nf2 
    hz = 2 * np.pi / nf3
    c = np.zeros(x.shape,dtype = fntau.dtype)
    device = cuda.get_current_device()
    n = x.shape[0] #number of kernels in the computing
    tpb = device.WARP_SIZE
    bpg = int(np.ceil(float(n)/tpb))
    gaussker_3d2_cuda[bpg, tpb](x, y, z, c, hx, hy, hz, nf1, nf2, nf3, nspread, tau, fntau)
    return c/(nf1*nf2*nf3)


"""
nufft1d type1, fast algrithm used FFT, convolution and decovolution
for mri recon, y is k-space data, x is the k-space trajectory, and output is image data
this is from https://jakevdp.github.io/blog/2015/02/24/optimizing-python-with-numpy-and-numba/
demo for using numba to accelerate the nufft

The Gaussian used for convolution is:
g(x) = exp(-x^2 / 4*tau) 
Fourier transform of g(x) is
G(k1) = exp(-k1^2*tau)

this is  computation of equation below
                  1  nj
     fk(k1,k2) = -- SUM cj(j) exp(+/-i k1*xj(j))
                 nj j=1

     for -ms/2 <= k1 <= (ms-1)/2,
inputs:
x is xj
c is coefficients 
ms is the length of output k1
df scaling factor on the k1 and k2; default is 1.0
iflag determine whether -1 or 1 sign for
exp(+/- i*k1*x); default is 1

output:
the nufft, output dim is ms X 1

"""
def nufft1d1_gaussker_cuda( x, c, ms, df=1.0, eps=1E-15, iflag=1, gridfast=0 ):
    """Fast Non-Uniform Fourier Transform with Numba"""
    nspread, nf1, tau = nufft_func._compute_1d_grid_params(ms, eps)

    if gridfast is 0:
        # Construct the convolved grid
        #ftau = nufft_func.build_grid_1d1(x * df, c, tau, nspread, np.zeros(nf1, dtype=c.dtype))
        ftau = build_grid_1d1_cuda2(x * df, c, tau, nspread, np.zeros(nf1, dtype=c.dtype))
    else:#fast griding with precomputing of some expoentials
        ftau = nufft_func.build_grid_1d1_fast(x * df, c, tau, nspread, np.zeros(nf1, dtype=c.dtype),\
                           np.zeros(nspread + 1, dtype=c.dtype))    

    # Compute the FFT on the convolved grid # do gpu fft later here
    if iflag < 0:
        Ftau = (1 / nf1) * np.fft.fft(ftau)
    else:
        Ftau = np.fft.ifft(ftau)
    #truncate the Ftau to match the size of output, alias are removed
    Ftau = np.concatenate([Ftau[-(ms//2):], Ftau[:ms//2 + ms % 2]])
    # Deconvolve the grid using convolution theorem, Ftau * G(k1)^-1
    k1 = nufft_func.nufftfreqs1d(ms)
    return (1 / len(x)) * np.sqrt(np.pi / tau) * np.exp(tau * k1 ** 2) * Ftau

#1d nufft type 2
def nufft1d2_gaussker_cuda( x, Fk, ms, df=1.0, eps=1E-15, iflag=1, gridfast=1 ):
    """Fast Non-Uniform Fourier Transform with Numba"""
    nspread, nf1, tau = nufft_func._compute_1d_grid_params(ms, eps)

    # Deconvolve the grid using convolution theorem, Ftau * G(k1)^-1
    k1 = nufft_func.nufftfreqs1d(ms)
    Fk = np.sqrt(np.pi / tau) * np.exp(tau * k1 ** 2) * Fk

    #reshape Fk and fftshift to match the size Fntau or Ftau
    Fntau = np.zeros(nf1, dtype=Fk.dtype)
    Fntau[-(ms//2):] = Fk[0:ms//2]
    Fntau[:ms//2 + ms % 2] = Fk[ms//2:ms]    

    # Compute the FFT on the convolved grid
    if iflag < 0:
        fntau = nf1 * np.fft.ifft(Fntau)
    else:
        fntau = np.fft.fft(Fntau)

    # Construct the convolved grid
    #if gridfast is not 1:
        #fx = build_grid_1d2(x/df, fntau, tau, nspread)
    #else:
        #fx = build_grid_1d2_fast(x/df, fntau, tau, nspread, np.zeros(nspread + 1, dtype=Fk.dtype))
    fx = build_grid_1d2_cuda(x/df, fntau, tau, nspread)
    return fx


"""
nufft2d type 1
The Gaussian used for convolution is:
g(x,y) = exp(-( x^2 + y^2 ) / 4*tau) 
Fourier transform of g(x) is
G(k1,k2) = exp(-( k1^2 + k2^2 )*tau)

this is computation of equation below
                  1  nj
     fk(k1,k2) = -- SUM cj(j) exp(+/-i*k1*xj(j)) exp(+/-i*k2*yj(j))
                 nj j=1

     for -ms/2 <= k1 <= (ms-1)/2, 
         -mt/2 <= k2 <= (mt-1)/2
inputs:
x is xj
y is yj
c is coefficients 
ms is the length of output k1
mt is the length of output k2
df scaling factor on the k1 and k2; default is 1.0
iflag determine whether -1 or 1 sign for 
exp(+/- i*k1*x) and  exp(+/- i*k2*y); default is 1

output:
the nufft, output dim is ms X mt

"""
def nufft2d1_gaussker_cuda( x, y, c, ms, mt, df=1.0, eps=1E-15, iflag=1, gridfast=1 ):
    """Fast Non-Uniform Fourier Transform with Numba"""
    nspread, nf1, nf2, tau = nufft_func._compute_2d_grid_params(ms, mt, eps)
    # Construct the convolved grid
    #ftau = build_grid_2d1(x * df, y * df, c, tau, nspread,
    #                  np.zeros((nf1, nf2), dtype = c.dtype))
    if gridfast is 0:
        ftau = nufft_func.build_grid_2d1(x * df, y * df, c, tau, nspread, np.zeros((nf1, nf2), dtype=c.dtype))
    else:#griding with precomputing of some exponentials
        ftau = nufft_func.build_grid_2d1_fast(x * df, y * df, c, tau, nspread, np.zeros((nf1, nf2), dtype=c.dtype),\
                           np.zeros((nspread + 1, nspread + 1), dtype=c.dtype))

    # Compute the FFT on the convolved grid
    if iflag < 0:
        Ftau = (1 / (nf1 * nf2)) * np.fft.fft2(ftau)
    else:
        Ftau = np.fft.ifft2(ftau) #1/(nf1*nf2) in ifft2 function when norm = None
    
    #truncate the Ftau to match the size of output, alias are removed
    Ftau = np.concatenate([Ftau[-(ms//2):,:], Ftau[:ms//2 + ms % 2,:]],0)    
    Ftau = np.concatenate([Ftau[:,-(mt//2):], Ftau[:,:mt//2 + mt % 2]],1)
    
    # Deconvolve the grid using convolution theorem, Ftau * G(k1,k2)^-1
    k1,k2 = nufft_func.nufftfreqs2d(ms, mt)
    # Note the np.sqrt(np.pi / tau)**2 due to the 2 dimentions of nufft
    return (1 / len(x)) * np.sqrt(np.pi / tau)**2 * np.exp(tau * (k1 ** 2 + k2 ** 2)) * Ftau # 

#2d nufft type 2
def nufft2d2_gaussker_cuda( x, y, Fk, ms, mt, df=1.0, eps=1E-15, iflag=1, gridfast=1 ):
    """Fast Non-Uniform Fourier Transform with Numba"""
    nspread, nf1, nf2, tau = nufft_func._compute_2d_grid_params(ms, mt, eps)

    # Deconvolve the grid using convolution theorem, Ftau * G(k1)^-1
    k1, k2 = nufft_func.nufftfreqs2d(ms, mt)
    # Note the np.sqrt(np.pi / tau)**2 due to the 2 dimentions of nufft
    Fk = np.sqrt(np.pi / tau)**2 * np.exp(tau * (k1 ** 2 + k2 ** 2)) * Fk #np.sqrt(np.pi / tau) * 

    #reshape Fk and fftshift to match the size Fntau or Ftau
    Fntau = np.zeros((nf1, nf2), dtype=Fk.dtype)
    Fntau[ -(ms//2):       ,       -(mt//2): ] = Fk[ 0:ms//2  ,  0:mt//2 ]#1 1
    Fntau[ :ms//2 + ms % 2 , :mt//2 + mt % 2 ] = Fk[ ms//2:ms , mt//2:mt ]#2 2    
    Fntau[ :ms//2 + ms % 2 ,       -(mt//2): ] = Fk[ ms//2:ms ,  0:mt//2 ]#2 1
    Fntau[ -(ms//2):       , :mt//2 + mt % 2 ] = Fk[ 0:ms//2  , mt//2:mt ]#1 2

    # Compute the FFT on the convolved grid
    if iflag < 0:
        fntau = nf1 * nf2 * np.fft.ifft2(Fntau)
    else:
        fntau = np.fft.fft2(Fntau)
    # Construct the convolved grid
    #if gridfast is not 1:
    #    fx = build_grid_2d2(x/df, y/df, fntau, tau, nspread)
    #else:
    #    fx = build_grid_2d2_fast(x/df, y/df, fntau, tau, nspread,\
    #        np.zeros((nspread + 1, nspread + 1), dtype=Fk.dtype))
    fx = build_grid_2d2_cuda(x/df, y/df, fntau, tau, nspread)
    return fx
"""
nufft3d type 1
The Gaussian used for convolution is:
g(x,y,z) = exp(-( x^2 + y^2 + z^2 ) / 4*tau) 
Fourier transform of g(x) is
G(k1,k2,k3) = exp(-( k1^2 + k2^2 + k3^2 )*tau)

this is computation of equation below
                  1  nj
     fk(k1,k2) = -- SUM cj(j) exp(+/-i k1*xj(j)) exp(+/-i k2*yj(j)) exp(+/-i k3*zj(j))
                 nj j=1

     for -ms/2 <= k1 <= (ms-1)/2, 
         -mt/2 <= k2 <= (mt-1)/2
         -mu/2 <= k3 <= (mu-1)/2
inputs:
x is xj
y is yj
z is zj
c is coefficients 
ms is the length of output k1
mt is the length of output k2
mu is the length of output k3
df scaling factor on the k1 and k2; default is 1.0
iflag determine whether -1 or 1 sign for exp(+/- i k x), 
exp(+/- i k2 y) and exp(+/- i k3 z); default is negative

output:
the nufft result, output dim is ms X mt X mu
"""
def nufft3d1_gaussker_cuda( x, y, z, c, ms, mt, mu, df=1.0, eps=1E-15, iflag=1, gridfast=2 ):
    """Fast Non-Uniform Fourier Transform with Numba"""
    nspread, nf1, nf2, nf3, tau = nufft_func._compute_3d_grid_params(ms, mt, mu, eps)
    #try to override nspread
    nspread = min(3, nspread)

    # Construct the convolved grid
    if gridfast is 0:
        ftau = nufft_func.build_grid_3d1(x * df, y * df, z *df, c, tau, nspread,\
                      np.zeros((nf1, nf2, nf3), dtype=c.dtype))
    else:#precompute some exponentials, not working
        ftau = nufft_func.build_grid_3d1_fast(x * df, y * df, z *df, c, tau, nspread,\
                      np.zeros((nf1, nf2, nf3), dtype=c.dtype), \
                      np.zeros((nspread+1, nspread+1, nspread+1), dtype=c.dtype)) 
      
    # Compute the FFT on the convolved grid
    if iflag < 0:
        Ftau = (1 / (nf1 * nf2 * nf3)) * np.fft.fftn(ftau,s=None,axes=(0,1,2))
    else:
        Ftau = np.fft.ifftn(ftau,s=None,axes=(0,1,2))
    #ut.plotim3(np.absolute(Ftau[:,:,:]))
    #truncate the Ftau to match the size of output, alias are removed
    Ftau = np.concatenate([Ftau[-(ms//2):,:,:], Ftau[:ms//2 + ms % 2,:,:]],0)    
    Ftau = np.concatenate([Ftau[:,-(mt//2):,:], Ftau[:,:mt//2 + mt % 2,:]],1)
    Ftau = np.concatenate([Ftau[:,:,-(mu//2):], Ftau[:,:,:mu//2 + mu % 2]],2)
    # Deconvolve the grid using convolution theorem, Ftau * G(k1,k2,k3)^-1
    k1, k2, k3 = nufft_func.nufftfreqs3d(ms, mt, mu)
    # Note the np.sqrt(np.pi / tau)**3 due to the 3 dimentions of nufft
    return (1 / len(x)) * np.sqrt(np.pi / tau)**3 * \
    np.exp(tau * (k1 ** 2 + k2 ** 2 + k3 ** 2)) * Ftau

#3d unfft type 2
def nufft3d2_gaussker_cuda( x, y, z, Fk, ms, mt, mu, df=1.0, eps=1E-15, iflag=1, gridfast=1 ):
    """Fast Non-Uniform Fourier Transform with Numba"""
    nspread, nf1, nf2, nf3, tau = nufft_func._compute_3d_grid_params(ms, mt, mu, eps)

    # Deconvolve the grid using convolution theorem, Ftau * G(k1)^-1
    k1, k2, k3 = nufft_func.nufftfreqs3d(ms, mt, mu)
    # Note the np.sqrt(np.pi / tau)**3 due to the 3 dimentions of nufft
    Fk = np.sqrt(np.pi / tau)**3 * np.exp(tau * (k1 ** 2 + k2 ** 2 + k3 ** 2)) * Fk

    #reshape Fk and fftshift to match the size Fntau or Ftau
    Fntau = np.zeros((nf1, nf2, nf3), dtype=Fk.dtype)
    Fntau[-(ms//2):      ,       -(mt//2):,       -(mu//2):] = Fk[0:ms//2 , 0:mt//2 , 0:mu//2]# 1 1 1
    Fntau[:ms//2 + ms % 2,       -(mt//2):,       -(mu//2):] = Fk[ms//2:ms, 0:mt//2 , 0:mu//2]# 2 1 1    
    Fntau[-(ms//2):      ,       -(mt//2):, :mu//2 + mu % 2] = Fk[0:ms//2 , 0:mt//2 ,mu//2:mu]# 1 1 2
    Fntau[-(ms//2):      , :mt//2 + mt % 2,       -(mu//2):] = Fk[0:ms//2 ,mt//2:mt , 0:mu//2]# 1 2 1
    Fntau[:ms//2 + ms % 2, :mt//2 + mt % 2,       -(mu//2):] = Fk[ms//2:ms,mt//2:mt , 0:mu//2]# 2 2 1
    Fntau[:ms//2 + ms % 2, :mt//2 + mt % 2, :mu//2 + mu % 2] = Fk[ms//2:ms,mt//2:mt ,mu//2:mu]# 2 2 2
    Fntau[:ms//2 + ms % 2,       -(mt//2):, :mu//2 + mu % 2] = Fk[ms//2:ms, 0:mt//2 ,mu//2:mu]# 2 1 2
    Fntau[-(ms//2):      , :mt//2 + mt % 2, :mu//2 + mu % 2] = Fk[0:ms//2 ,mt//2:mt ,mu//2:mu]# 1 2 2

    # Compute the FFT on the convolved grid
    if iflag < 0:
        fntau = (nf1 * nf2 * nf3) * np.fft.ifftn(Fntau,s=None,axes=(0,1,2))
    else:
        fntau = np.fft.fftn(Fntau,s=None,axes=(0,1,2))

    # Construct the convolved grid
    #if gridfast is not 1:
    #    fx = build_grid_3d2(x/df, y/df, z/df, fntau, tau, nspread)
    #else:
    #    fx = build_grid_3d2_fast(x/df, y/df, z/df, fntau, tau, nspread,\
    #     np.zeros((nspread+1, nspread+1, nspread+1), dtype=Fk.dtype))
    fx = build_grid_3d2_cuda(x/df, y/df, z/df, fntau, tau, nspread)
    return fx

if __name__ == "__main__":
    
    #test nufft type1
    #nufft_func.time_nufft1d1(nufft1d1_gaussker_cuda,64,5120,5)
    #nufft_func.time_nufft2d1(nufft2d1_gaussker_cuda,64,64,5120)
    #nufft_func.time_nufft3d1(nufft3d1_gaussker_cuda,32,32,16,2048)
    
    #test nufft type2
    #nufft_func.time_nufft1d2(nufft1d1_gaussker_cuda,nufft1d2_gaussker_cuda,32,102400,10)
    #nufft_func.time_nufft2d2(nufft2d1_gaussker_cuda,nufft2d2_gaussker_cuda,16,16,25000,1)
    #nufft_func.time_nufft3d2(nufft3d1_gaussker_cuda,nufft3d2_gaussker_cuda,8,8,8,204800,1)

    #compare
    nufft_func.compare_nufft1d1(nufft1d1_gaussker_cuda,32,3200)
    #nufft_func.compare_nufft2d1(nufft2d1_gaussker_cuda, 64, 64,2500)
    #nufft_func.compare_nufft3d1(nufft3d1_gaussker_cuda, 32, 32,16,2048)
