from __future__ import print_function, division
import numpy as np
import numba
import utilities.utilities_func as ut
import nufft_func
import nufft_test_func
from numba import cuda
from time import time
from math import exp
from fft.cufft import fftnc2c_cuda, ifftnc2c_cuda


#####################################################################################################
#unfft 1d, type1, type2 and type2&type1 (AHA)
#####################################################################################################
#type 1
#kernal for griding in cuda
#I used atom_add to solve the memory conflict in gpu
#@cuda.jit('void(float64[:], complex128[:], float64, int32, int32, float64, complex128[:])')
@cuda.jit
def gaussker_1d1_cuda(x, c, hx, nf1, nspread, tau, real_ftau, imag_ftau ):
    """This kernel function for gauss grid 1d typ1, and it will be executed by a thread."""
    i  = cuda.grid(1)
    if i > x.shape[0]:
        return
    #do the 1d griding here
    xi = x[i] % (2 * np.pi) #x, shift the source point xj so that it lies in [0,2*pi]
    mx = 1 + int(xi // hx) #index for the closest grid point
    for mmx in range(-nspread, nspread): #mm index for all the spreading points
        #griding with g(x) = exp(-(x^2) / 4*tau)
        #ftau[(mx + mmx) % nf1] += c[i] * exp(-0.25 * ((xi - hx * (mx + mmx)) ** 2 ) / tau)
        tmp = c[i] * exp(-0.25 * ((xi - hx * (mx + mmx)) ** 2 ) / tau)
        cuda.atomic.add(real_ftau, (mx + mmx) % nf1, tmp.real)
        cuda.atomic.add(imag_ftau, (mx + mmx) % nf1, tmp.imag)

# do griding with cuda acceleration
def build_grid_1d1_cuda( x, c, tau, nspread, ftau ):
    nf1 = ftau.shape[0]
    hx = 2 * np.pi / nf1
    device = cuda.get_current_device()
    n = x.shape[0] #number of kernels in the computing
    tpb = device.WARP_SIZE
    bpg = int(np.ceil(float(n)/tpb))
    real_ftau = np.zeros(ftau.shape,np.float64)
    imag_ftau = np.zeros(ftau.shape,np.float64)
    gaussker_1d1_cuda[bpg, tpb](x, c, hx, nf1, nspread, tau, real_ftau, imag_ftau)
    ftau = real_ftau + 1j*imag_ftau
    return ftau

#type1, fast version with precompute of exponentials
@cuda.jit
def gaussker_1d1_fast_cuda(x, c, hx, nf1, nspread, tau, E3, real_ftau, imag_ftau ):
    """This kernel function for gauss grid 1d typ1, and it will be executed by a thread."""
    i  = cuda.grid(1)
    if i > x.shape[0]:
        return
    xi = x[i] % (2 * np.pi) #x
    mx = 1 + int(xi // hx) #index for the closest grid point
    xi = (xi - hx * mx) #
    E1 = exp(-0.25 * xi ** 2 / tau)
    E2 = exp((xi * np.pi) / (nf1 * tau))
    E2mm = 1
    for mmx in range(nspread):
        #ftau[(m + mm) % nf1] +
        tmpp = c[i] * E1 * E2mm * E3[mmx]
        E2mm *= E2
        #ftau[(m - mm - 1) % nf1] +
        tmpn = c[i] * E1 / E2mm * E3[mmx + 1]
        cuda.atomic.add(real_ftau, (mx + mmx) % nf1,     tmpp.real)
        cuda.atomic.add(imag_ftau, (mx + mmx) % nf1,     tmpp.imag)
        cuda.atomic.add(real_ftau, (mx - mmx - 1) % nf1, tmpn.real)
        cuda.atomic.add(imag_ftau, (mx - mmx - 1) % nf1, tmpn.imag)

def build_grid_1d1_fast_cuda( x, c, tau, nspread, ftau, E3 ):
    nf1 = ftau.shape[0]
    hx = 2 * np.pi / nf1
    # precompute some exponents
    for j in range(nspread + 1):
        E3[j] = exp(-(np.pi * j / nf1) ** 2 / tau)
    device = cuda.get_current_device()
    n = x.shape[0] #number of kernels in the computing
    tpb = device.WARP_SIZE
    bpg = int(np.ceil(float(n)/tpb))
    real_ftau = np.zeros(ftau.shape,np.float64)
    imag_ftau = np.zeros(ftau.shape,np.float64)
    gaussker_1d1_fast_cuda[bpg, tpb](x, c, hx, nf1, nspread, tau, E3, real_ftau, imag_ftau)
    ftau = real_ftau + 1j*imag_ftau
    return ftau

# type 2, kernal for griding in cuda
@cuda.jit
def gaussker_1d2_cuda(x, c, hx, nf1, nspread, tau, fntau ):
    """This kernel function for gauss grid 1d typ1, and it will be executed by a thread."""
    i  = cuda.grid(1)
    if i > x.shape[0]:
        return
    #do the 1d griding here
    xi = x[i] % (2 * np.pi) #x, shift the source point xj so that it lies in [0,2*pi]
    mx = 1 + int(xi // hx) #index for the closest grid point
    for mmx in range(-nspread, nspread): #mm index for all the spreading points
        #griding with g(x,y) = exp(-(x^2 + y^2) / 4*tau)
        c[i] += fntau[(mx + mmx) % nf1] * exp(-0.25 * ((xi - hx * (mx + mmx)) ** 2 ) / tau)

# do griding with cuda acceleration
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

#fast version with precompute of exponentials
@cuda.jit
def gaussker_1d2_fast_cuda( x, c, hx, nf1, nspread, tau, E3, fntau ):
    """This kernel function for gauss grid 1d typ1, and it will be executed by a thread."""
    i  = cuda.grid(1)
    if i > x.shape[0]:
        return
    xi = x[i] % (2 * np.pi) #x
    mx = 1 + int(xi // hx) #index for the closest grid point
    xi = (xi - hx * mx) #
    E1 = exp(-0.25 * xi ** 2 / tau)
    E2 = exp((xi * np.pi) / (nf1 * tau))
    E2mm = 1
    for mmx in range(nspread):
        c[i] += fntau[(mx + mmx) % nf1] * E1 * E2mm * E3[mmx]
        E2mm *= E2
        c[i] += fntau[(mx - mmx - 1) % nf1] * E1 / E2mm * E3[mmx + 1]

def build_grid_1d2_fast_cuda( x, fntau, tau, nspread, E3 ):
    nf1 = fntau.shape[0]
    hx = 2 * np.pi / nf1
    c = np.zeros(x.shape,dtype=fntau.dtype)
    # precompute some exponentials
    for j in range(nspread + 1):
        E3[j] = exp(-(np.pi * j / nf1) ** 2 / tau)
    device = cuda.get_current_device()
    n = x.shape[0] #number of kernels in the computing
    tpb = device.WARP_SIZE
    bpg = int(np.ceil(float(n)/tpb))
    gaussker_1d2_fast_cuda[bpg, tpb](x, c, hx, nf1, nspread, tau, E3, fntau)
    return c/nf1

#type 2 & type 1
#input fntau, output real/imag_ftau
@cuda.jit
def gaussker_1d21_cuda(x, hx, nf1, nspread, tau, fntau, real_ftau, imag_ftau ):
    """This kernel function for gauss grid 3d type2, and it will be executed by a thread."""
    i  = cuda.grid(1)
    if i > x.shape[0]:
        return

    xi = x[i] % (2.0 * np.pi) #x, shift the source point xj so that it lies in [0,2*pi]
    mx = 1 + int(xi // hx) #index for the closest grid point
    c  = 0.0 #coefficient, saved temporarily
    #do the 3d griding here
    for mmx in range(-nspread, nspread): #mm index for all the spreading points
        #griding with g(x,y) = exp(-(x^2 + y^2 ) / 4*tau)
        c += fntau[(mx + mmx) % nf1] * exp(-0.25 * ((xi - hx * (mx + mmx)) ** 2 ) / tau)
    #type2 nufft has 1/N term
    c = c/(nf1) 
    #do grid again
    for mmx in range(-nspread, nspread): #mm index for all the spreading points
        #griding with g(x,y) = exp(-(x^2 + y^2) / 4*tau)
        tmp = c * exp(-0.25 * ((xi - hx * (mx + mmx)) ** 2 ) / tau)
        cuda.atomic.add(real_ftau, ((mx + mmx) % nf1), tmp.real)
        cuda.atomic.add(imag_ftau, ((mx + mmx) % nf1), tmp.imag)
 
def build_grid_1d21_cuda( x, ftau, tau, nspread ):
    #number of pioints along x, y, z
    nf1 = ftau.shape[0]
    #minimal intervals along x, y, z
    hx = 2 * np.pi / nf1
    #c is coefficients
    #c = np.zeros(x.shape,dtype = ftau.dtype)
    #prepare for CUDA, compute CUDA parameters n, tpb, bpg
    device = cuda.get_current_device()
    n = x.shape[0] #number of kernels in the computing
    tpb = device.WARP_SIZE
    bpg = int(np.ceil(float(n)/tpb))
    real_ftau = np.zeros(ftau.shape, dtype=np.float64)
    imag_ftau = np.zeros(ftau.shape, dtype=np.float64) #atom add only support float32 or 64
    #computing start here
    gaussker_1d21_cuda[bpg, tpb](x, hx, nf1, nspread, tau, ftau, real_ftau, imag_ftau)
    ftau = real_ftau + 1j*imag_ftau
    return ftau


"""
main function of nufft1d, fast algrithm used FFT, convolution and decovolution
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
#1d nufft type 1 
def nufft1d1_gaussker_cuda( x, c, ms, df=1.0, eps=1E-15, iflag=1, gridfast=0 ):
    """Fast Non-Uniform Fourier Transform with Numba"""
    nspread, nf1, tau = nufft_func._compute_1d_grid_params(ms, eps)

    if gridfast is 0:
        # Construct the convolved grid
        #Ftau = nufft_func.build_grid_1d1(x * df, c, tau, nspread, np.zeros(nf1, dtype=c.dtype))
        Ftau = build_grid_1d1_cuda(x * df, c, tau, nspread, np.zeros(nf1, dtype=c.dtype))
    else:#fast griding with precomputing of some expoentials
        Ftau = build_grid_1d1_fast_cuda(x * df, c, tau, nspread, np.zeros(nf1, dtype=c.dtype),\
                           np.zeros(nspread + 1, dtype=c.dtype))

    # Compute the FFT on the convolved grid # do gpu fft later here
    if iflag < 0:
        Ftau = (1 / nf1) * fftnc2c_cuda(Ftau)#np.fft.fft(Ftau)#
    else:
        Ftau = ifftnc2c_cuda(Ftau)#np.fft.ifft(Ftau)#
    #truncate the Ftau to match the size of output, alias are removed
    Ftau = np.concatenate([Ftau[-(ms//2):], Ftau[:ms//2 + ms % 2]])
    # Deconvolve the grid using convolution theorem, Ftau * G(k1)^-1
    k1 = nufft_func.nufftfreqs1d(ms)
    return (1 / len(x)) * np.sqrt(np.pi / tau) * np.exp(tau * k1 ** 2) * Ftau

#1d nufft type 2
def nufft1d2_gaussker_cuda( x, Fk, ms, df=1.0, eps=1E-15, iflag=1, gridfast=0 ):
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
        Fntau = nf1 * ifftnc2c_cuda(Fntau)#np.fft.ifft(Fntau)#
    else:
        Fntau = fftnc2c_cuda(Fntau)#np.fft.fft(Fntau)#

    # Construct the convolved grid
    if gridfast is not 1:
        fx = build_grid_1d2_cuda(x * df, Fntau, tau, nspread)
    else:
        fx = build_grid_1d2_fast_cuda(x * df, Fntau, tau, nspread, np.zeros(nspread + 1, dtype=Fk.dtype))
    
    return fx

#1d nufft type 2 & type 1
def nufft1d21_gaussker_cuda( x, Fk, ms, df=1.0, eps=1E-15, iflag=1, gridfast=0 ):
    """Fast Non-Uniform Fourier Transform with Numba"""
    nspread, nf1, tau = nufft_func._compute_1d_grid_params(ms, eps)

    # Deconvolve the grid using convolution theorem, Ftau * G(k1)^-1
    k1 = nufft_func.nufftfreqs1d(ms)
    Fk = np.sqrt(np.pi / tau) * np.exp(tau * k1 ** 2) * Fk

    #reshape Fk and fftshift to match the size Ftau 
    Ftau = np.zeros(nf1, dtype=Fk.dtype)
    Ftau[-(ms//2):] = Fk[0:ms//2]
    Ftau[:ms//2 + ms % 2] = Fk[ms//2:ms]

    # Compute the FFT on the convolved grid
    if iflag < 0:
        Ftau = nf1 * ifftnc2c_cuda(Ftau)#np.fft.ifft(Ftau)#
    else:
        Ftau = fftnc2c_cuda(Ftau)#np.fft.fft(Ftau)#

    # Construct the convolved grid
    if 1: #gridfast is not 1:
        Ftau = build_grid_1d21_cuda(x * df, Ftau, tau, nspread)
    #else:
    #    Ftau = build_grid_1d21_fast_cuda(x * df, Ftau, tau, nspread, np.zeros(nspread + 1, dtype=Fk.dtype))

    # Compute the FFT on the convolved grid # do gpu fft later here
    if iflag < 0:
        Ftau = (1 / nf1) * fftnc2c_cuda(Ftau)#np.fft.fft(Ftau)#
    else:
        Ftau = ifftnc2c_cuda(Ftau)#np.fft.ifft(Ftau)#
    #truncate the Ftau to match the size of output, alias are removed
    Ftau = np.concatenate([Ftau[-(ms//2):], Ftau[:ms//2 + ms % 2]])
    # Deconvolve the grid using convolution theorem, Ftau * G(k1)^-1
    return (1 / len(x)) * np.sqrt(np.pi / tau) * np.exp(tau * k1 ** 2) * Ftau

#####################################################################################################
#unfft 2d, type1, type2 and type2&type1 (AHA)
#####################################################################################################
# type 1
# kernal for griding in cuda, atom add is used to solve the momory conflict
@cuda.jit
def gaussker_2d1_cuda(x, y, c, hx, hy, nf1, nf2, nspread, tau, real_ftau, imag_ftau ):
    """This kernel function for gauss grid 1d typ1, and it will be executed by a thread."""
    i  = cuda.grid(1)
    if i > x.shape[0]:
        return
    #do the 1d griding here
    xi = x[i] % (2 * np.pi) #x, shift the source point xj so that it lies in [0,2*pi]
    yi = y[i] % (2 * np.pi) #y, shift the source point yj so that it lies in [0,2*pi]
    mx = 1 + int(xi // hx) #index for the closest grid point
    my = 1 + int(yi // hy) #index for the closest grid point
    for mmx in range(-nspread, nspread): #mm index for all the spreading points
        for mmy in range(-nspread,nspread):
            #griding with g(x,y) = exp(-(x^2 + y^2) / 4*tau)
            #ftau[(mx + mmx) % nf1, (my + mmy) % nf2] +=
            tmp = c[i] * exp(-0.25 * (\
            (xi - hx * (mx + mmx)) ** 2 + \
            (yi - hy * (my + mmy)) ** 2 ) / tau)
            cuda.atomic.add(real_ftau, ((mx + mmx) % nf1, (my + mmy) % nf2), tmp.real)
            cuda.atomic.add(imag_ftau, ((mx + mmx) % nf1, (my + mmy) % nf2), tmp.imag)

def build_grid_2d1_cuda( x, y, c, tau, nspread, ftau ):
    nf1 = ftau.shape[0]
    nf2 = ftau.shape[1]
    hx = 2 * np.pi / nf1
    hy = 2 * np.pi / nf2
    device = cuda.get_current_device()
    n = x.shape[0] #number of kernels in the computing
    tpb = device.WARP_SIZE
    bpg = int(np.ceil(float(n)/tpb))
    real_ftau = np.zeros(ftau.shape, dtype=np.float64)
    imag_ftau = np.zeros(ftau.shape, dtype=np.float64) #atom add only support float32 or 64
    gaussker_2d1_cuda[bpg, tpb](x, y, c, hx, hy, nf1, nf2, nspread, tau, real_ftau, imag_ftau)
    ftau = real_ftau + 1j*imag_ftau
    return ftau

#type 1, fast version with precomputation of exponentials
@cuda.jit
def gaussker_2d1_fast_cuda(x, y, c, hx, hy, nf1, nf2, nspread, tau, E3, real_ftau, imag_ftau ):
    """This kernel function for gauss grid 1d typ1, and it will be executed by a thread."""
    i     = cuda.grid(1)
    if i > x.shape[0]:
        return
    xi    = x[i] % (2 * np.pi) #x
    yi    = y[i] % (2 * np.pi) #y
    mx    = 1 + int(xi // hx) #index for the closest grid point
    my    = 1 + int(yi // hy)
    xi    = (xi - hx * mx) #
    yi    = (yi - hy * my)
    E1    = exp(-0.25 * (xi ** 2 + yi ** 2) / tau)
    E2x   = exp((xi * np.pi) / (nf1 * tau))
    E2y   = exp((yi * np.pi) / (nf2 * tau))
    E2mmx = 1
    V0    = c[i] * E1
    for mmx in range(nspread):
        E2mmy = 1
        for mmy in range(nspread):#use the symmetry of E1, E2 and E3, e.g. 1/(E2(mmx)*E2x) = E2(mx-mmx)
            #ftau[    (mx + mmx) % nf1,     (my + mmy) % nf2] +=
            tmpxpyp = V0 * E2mmx       * E2mmy       * E3[    mmx,     mmy]
            #ftau[    (mx + mmx) % nf1, (my - mmy - 1) % nf2] +=
            tmpxpyn = V0 * E2mmx       / (E2mmy*E2y) * E3[    mmx, mmy + 1]
            #ftau[(mx - mmx - 1) % nf1,     (my + mmy) % nf2] +=
            tmpxnyp = V0 / (E2mmx*E2x) * E2mmy       * E3[mmx + 1,     mmy]
            #ftau[(mx - mmx - 1) % nf1, (my - mmy - 1) % nf2] +=
            tmpxnyn = V0 / (E2mmx*E2x) / (E2mmy*E2y) * E3[mmx + 1, mmy + 1]
            #atom add solves the memory conflict issue in GPU
            cuda.atomic.add(real_ftau, (    (mx + mmx) % nf1,     (my + mmy) % nf2), tmpxpyp.real) #x  1, y  1
            cuda.atomic.add(imag_ftau, (    (mx + mmx) % nf1,     (my + mmy) % nf2), tmpxpyp.imag)
            cuda.atomic.add(real_ftau, (    (mx + mmx) % nf1, (my - mmy - 1) % nf2), tmpxpyn.real) #x  1, y -1
            cuda.atomic.add(imag_ftau, (    (mx + mmx) % nf1, (my - mmy - 1) % nf2), tmpxpyn.imag)
            cuda.atomic.add(real_ftau, ((mx - mmx - 1) % nf1,     (my + mmy) % nf2), tmpxnyp.real) #x -1, y  1
            cuda.atomic.add(imag_ftau, ((mx - mmx - 1) % nf1,     (my + mmy) % nf2), tmpxnyp.imag)
            cuda.atomic.add(real_ftau, ((mx - mmx - 1) % nf1, (my - mmy - 1) % nf2), tmpxnyn.real) #x -1, y -1
            cuda.atomic.add(imag_ftau, ((mx - mmx - 1) % nf1, (my - mmy - 1) % nf2), tmpxnyn.imag)
            E2mmy *= E2y
        E2mmx *= E2x

def build_grid_2d1_fast_cuda( x, y, c, tau, nspread, ftau, E3 ):
    nf1 = ftau.shape[0]
    nf2 = ftau.shape[1]
    hx  = 2 * np.pi / nf1
    hy  = 2 * np.pi / nf2
    # precompute some exponents
    for l in range(nspread + 1):
        for j in range(nspread + 1):
            E3[j,l] = exp(-((np.pi * j / nf1) ** 2 + (np.pi * l /nf2) ** 2)/ tau)
    device    = cuda.get_current_device()
    n         = x.shape[0] #number of kernels in the computing
    tpb       = device.WARP_SIZE
    bpg       = int(np.ceil(float(n)/tpb))
    real_ftau = np.zeros(ftau.shape,np.float64)
    imag_ftau = np.zeros(ftau.shape,np.float64)
    gaussker_2d1_fast_cuda[bpg, tpb](x, y, c, hx, hy, nf1, nf2, nspread, tau, E3, real_ftau, imag_ftau )
    ftau = real_ftau + 1j*imag_ftau
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
    mx = 1 + int(xi // hx) #index for the closest grid point
    my = 1 + int(yi // hy) #index for the closest grid point
    for mmx in range(-nspread, nspread): #mm index for all the spreading points
        for mmy in range(-nspread,nspread):
            #griding with g(x,y) = exp(-(x^2 + y^2) / 4*tau)
            c[i] \
            += fntau[(mx + mmx) % nf1, (my + mmy) % nf2] * exp(-0.25 * (\
            (xi - hx * (mx + mmx)) ** 2 + \
            (yi - hy * (my + mmy)) ** 2 ) / tau)

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

#type 2
#kernal for griding in cuda
#fast version with precompute of exponentials
@cuda.jit
def gaussker_2d2_fast_cuda(x, y, c, hx, hy, nf1, nf2, nspread, tau, E3, fntau ):
    """This kernel function for gauss grid 1d typ1, and it will be executed by a thread."""
    i  = cuda.grid(1)
    if i > x.shape[0]:
        return
    #do the 2d griding here
    xi = x[i] % (2 * np.pi) #x
    yi = y[i] % (2 * np.pi) #y
    mx = 1 + int(xi // hx) #index for the closest grid point
    my = 1 + int(yi // hy)
    xi = (xi - hx * mx) #
    yi = (yi - hy * my)
    E1 = exp(-0.25 * (xi ** 2 + yi ** 2) / tau)
    E2x = exp((xi * np.pi) / (nf1 * tau))
    E2y = exp((yi * np.pi) / (nf2 * tau))
    E2mmx = 1
    for mmx in range(nspread):
        E2mmy = 1
        for mmy in range(nspread):#use the symmetry of E1, E2 and E3, e.g. 1/(E2(mmx)*E2x) = E2(mx-mmx)
            c[i] += fntau[    (mx + mmx) % nf1,     (my + mmy) % nf2] * E1 * E2mmx       * E2mmy       * E3[    mmx,     mmy]
            c[i] += fntau[    (mx + mmx) % nf1, (my - mmy - 1) % nf2] * E1 * E2mmx       / (E2mmy*E2y) * E3[    mmx, mmy + 1]
            c[i] += fntau[(mx - mmx - 1) % nf1,     (my + mmy) % nf2] * E1 / (E2mmx*E2x) * E2mmy       * E3[mmx + 1,     mmy]
            c[i] += fntau[(mx - mmx - 1) % nf1, (my - mmy - 1) % nf2] * E1 / (E2mmx*E2x) / (E2mmy*E2y) * E3[mmx + 1, mmy + 1]
            E2mmy *= E2y
        E2mmx *= E2x

def build_grid_2d2_fast_cuda( x, y, fntau, tau, nspread, E3 ):
    nf1 = fntau.shape[0]
    nf2 = fntau.shape[1]
    hx = 2 * np.pi / nf1
    hy = 2 * np.pi / nf2
    c = np.zeros(x.shape,dtype = fntau.dtype)
    # precompute some exponents
    for l in range(nspread + 1):
        for j in range(nspread + 1):
            E3[j,l] = exp(-((np.pi * j / nf1) ** 2 + (np.pi * l /nf2) ** 2)/ tau)
    device = cuda.get_current_device()
    n = x.shape[0] #number of kernels in the computing
    tpb = device.WARP_SIZE
    bpg = int(np.ceil(float(n)/tpb))
    gaussker_2d2_fast_cuda[bpg, tpb](x, y, c, hx, hy, nf1, nf2, nspread, tau, E3, fntau )
    return c/(nf1*nf2)

#type 2 & type 1
#input fntau, output real/imag_ftau
@cuda.jit
def gaussker_2d21_cuda(x, y, hx, hy, nf1, nf2, nspread, tau, fntau, real_ftau, imag_ftau ):
    """This kernel function for gauss grid 3d type2, and it will be executed by a thread."""
    i  = cuda.grid(1)
    if i > x.shape[0]:
        return

    xi = x[i] % (2.0 * np.pi) #x, shift the source point xj so that it lies in [0,2*pi]
    yi = y[i] % (2.0 * np.pi) #y, shift the source point yj so that it lies in [0,2*pi]
    mx = 1 + int(xi // hx) #index for the closest grid point
    my = 1 + int(yi // hy) #index for the closest grid point
    c  = 0.0#coefficient, saved temporarily
    #do the 3d griding here
    for mmx in range(-nspread, nspread): #mm index for all the spreading points
        for mmy in range(-nspread,nspread):#mm index for all the spreading points
            #griding with g(x,y) = exp(-(x^2 + y^2 ) / 4*tau)
            c += fntau[(mx + mmx) % nf1, (my + mmy) % nf2] * exp(-0.25 * (\
            (xi - hx * (mx + mmx)) ** 2 + \
            (yi - hy * (my + mmy)) ** 2 ) / tau)
    #type2 nufft has 1/N term
    c = c/(nf1*nf2) 
    #grid again
    for mmx in range(-nspread, nspread): #mm index for all the spreading points
        for mmy in range(-nspread,nspread):
            #griding with g(x,y) = exp(-(x^2 + y^2) / 4*tau)
            tmp = c * exp(-0.25 * (\
            (xi - hx * (mx + mmx)) ** 2 + \
            (yi - hy * (my + mmy)) ** 2 ) / tau)
            cuda.atomic.add(real_ftau, ((mx + mmx) % nf1, (my + mmy) % nf2), tmp.real)
            cuda.atomic.add(imag_ftau, ((mx + mmx) % nf1, (my + mmy) % nf2), tmp.imag)

def build_grid_2d21_cuda( x, y, ftau, tau, nspread ):
    #number of pioints along x, y, z
    nf1 = ftau.shape[0]
    nf2 = ftau.shape[1]
    #minimal intervals along x, y, z
    hx = 2 * np.pi / nf1
    hy = 2 * np.pi / nf2
    #c is coefficients
    #c = np.zeros(x.shape,dtype = ftau.dtype)
    #prepare for CUDA, compute CUDA parameters n, tpb, bpg
    device = cuda.get_current_device()
    n = x.shape[0] #number of kernels in the computing
    tpb = device.WARP_SIZE
    bpg = int(np.ceil(float(n)/tpb))
    real_ftau = np.zeros(ftau.shape, dtype=np.float64)
    imag_ftau = np.zeros(ftau.shape, dtype=np.float64) #atom add only support float32 or 64
    #computing start here
    gaussker_2d21_cuda[bpg, tpb](x, y, hx, hy, nf1, nf2, nspread, tau, ftau, real_ftau, imag_ftau)
    ftau = real_ftau + 1j*imag_ftau
    return ftau

"""
main function of nufft2d
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
# 2d nufft type 1
def nufft2d1_gaussker_cuda( x, y, c, ms, mt, df=1.0, eps=1E-15, iflag=1, gridfast=1 ):
    """Fast Non-Uniform Fourier Transform with Numba"""
    nspread, nf1, nf2, tau = nufft_func._compute_2d_grid_params(ms, mt, eps)
    # Construct the convolved grid
    #Ftau = build_grid_2d1(x * df, y * df, c, tau, nspread,
    #                  np.zeros((nf1, nf2), dtype = c.dtype))
    if gridfast is 0:
        ftau = build_grid_2d1_cuda(x * df, y * df, c, tau, nspread, np.zeros((nf1, nf2), dtype=c.dtype))
    else:#griding with precomputing of some exponentials
        ftau = build_grid_2d1_fast_cuda(x * df, y * df, c, tau, nspread, np.zeros((nf1, nf2), dtype=c.dtype),\
                           np.zeros((nspread + 1, nspread + 1), dtype=c.dtype))

    # Compute the FFT on the convolved grid
    if iflag < 0:
        Ftau = (1 / (nf1 * nf2)) * fftnc2c_cuda(ftau)#np.fft.fft2(Ftau)#
    else:
        Ftau = ifftnc2c_cuda(ftau)#np.fft.ifft2(Ftau) #

    #truncate the Ftau to match the size of output, alias are removed
    Ftau = np.concatenate([Ftau[-(ms//2):,:], Ftau[:ms//2 + ms % 2,:]],0)
    Ftau = np.concatenate([Ftau[:,-(mt//2):], Ftau[:,:mt//2 + mt % 2]],1)

    # Deconvolve the grid using convolution theorem, Ftau * G(k1,k2)^-1
    k1,k2 = nufft_func.nufftfreqs2d(ms, mt)
    # Note the np.sqrt(np.pi / tau)**2 due to the 2 dimentions of nufft
    return (1 / len(x)) * np.sqrt(np.pi / tau)**2 * np.exp(tau * (k1 ** 2 + k2 ** 2)) * Ftau #

#2d nufft type 2
def nufft2d2_gaussker_cuda( x, y, Fk, ms, mt, df=1.0, eps=1E-15, iflag=1, gridfast=0 ):
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
        Fntau = nf1 * nf2 * ifftnc2c_cuda(Fntau)#np.fft.ifft2(Fntau)#
    else:
        Fntau = fftnc2c_cuda(Fntau)#np.fft.fft2(Fntau)#
    # Construct the convolved grid
    if gridfast is not 1:
        fx = build_grid_2d2_cuda(x * df, y * df, Fntau, tau, nspread)
    else:
        fx = build_grid_2d2_fast_cuda(x * df, y * df, Fntau, tau, nspread,\
            np.zeros((nspread + 1, nspread + 1), dtype=Fk.dtype))
    return fx

#2d nufft type 2 & type1
def nufft2d21_gaussker_cuda( x, y, Fk, ms, mt, df=1.0, eps=1E-15, iflag=1, gridfast=0 ):
    """Fast Non-Uniform Fourier Transform with Numba"""
    nspread, nf1, nf2, tau = nufft_func._compute_2d_grid_params(ms, mt, eps)

    # Deconvolve the grid using convolution theorem, Ftau * G(k1)^-1
    k1, k2 = nufft_func.nufftfreqs2d(ms, mt)
    # Note the np.sqrt(np.pi / tau)**2 due to the 2 dimentions of nufft
    Fk = np.sqrt(np.pi / tau)**2 * np.exp(tau * (k1 ** 2 + k2 ** 2)) * Fk #np.sqrt(np.pi / tau) *

    #reshape Fk and fftshift to match the size Ftau
    Ftau = np.zeros((nf1, nf2), dtype=Fk.dtype)
    Ftau[ -(ms//2):       ,       -(mt//2): ] = Fk[ 0:ms//2  ,  0:mt//2 ]#1 1
    Ftau[ :ms//2 + ms % 2 , :mt//2 + mt % 2 ] = Fk[ ms//2:ms , mt//2:mt ]#2 2
    Ftau[ :ms//2 + ms % 2 ,       -(mt//2): ] = Fk[ ms//2:ms ,  0:mt//2 ]#2 1
    Ftau[ -(ms//2):       , :mt//2 + mt % 2 ] = Fk[ 0:ms//2  , mt//2:mt ]#1 2

    # Compute the FFT on the convolved grid
    if iflag < 0:
        Ftau = nf1 * nf2 * ifftnc2c_cuda(Ftau)#np.fft.ifft2(Ftau)#
    else:
        Ftau = fftnc2c_cuda(Ftau)#np.fft.fft2(Ftau)#
    # Construct the convolved grid
    if 1:#gridfast is not 1:
        Ftau = build_grid_2d21_cuda(x * df, y * df, Ftau, tau, nspread)
    #else:
        #Ftau = build_grid_2d21_fast_cuda(x, y, Ftau, tau, nspread,\
        #    np.zeros((nspread + 1, nspread + 1), dtype=Fk.dtype))

    # Compute the FFT on the convolved grid
    if iflag < 0:
        Ftau = (1 / (nf1 * nf2)) * fftnc2c_cuda(Ftau)#np.fft.fft2(Ftau)#
    else:
        Ftau = ifftnc2c_cuda(Ftau)#np.fft.ifft2(Ftau) #

    #truncate the Ftau to match the size of output, alias are removed
    Ftau = np.concatenate([Ftau[-(ms//2):,:], Ftau[:ms//2 + ms % 2,:]],0)
    Ftau = np.concatenate([Ftau[:,-(mt//2):], Ftau[:,:mt//2 + mt % 2]],1)

    # Deconvolve the grid using convolution theorem, Ftau * G(k1,k2)^-1
    # Note the np.sqrt(np.pi / tau)**2 due to the 2 dimentions of nufft
    return (1 / len(x)) * np.sqrt(np.pi / tau)**2 * np.exp(tau * (k1 ** 2 + k2 ** 2)) * Ftau #

#####################################################################################################
#unfft 3d, type1, type2 and type2&type1 (AHA)
#####################################################################################################
#type 1,kernal for griding in cuda
#@cuda.jit('void(float64[:], complex128[:], float64, int32, int32, float64, complex128[:])')
@cuda.jit
def gaussker_3d1_cuda(x, y, z, c, hx, hy, hz, nf1, nf2, nf3, nspread, tau, real_ftau, imag_ftau ):
    """This kernel function for gauss grid 1d typ1, and it will be executed by a thread."""
    i  = cuda.grid(1)
    if i > x.shape[0]:
        return
    #do the 1d griding here
    xi = x[i] % (2.0 * np.pi) #x, shift the source point xj so that it lies in [0,2*pi]
    yi = y[i] % (2.0 * np.pi) #y, shift the source point yj so that it lies in [0,2*pi]
    zi = z[i] % (2.0 * np.pi) #z, shift the source point zj so that it lies in [0,2*pi]
    mx = 1 + int(xi // hx) #index for the closest grid point
    my = 1 + int(yi // hy) #index for the closest grid point
    mz = 1 + int(zi // hz) #index for the closest grid point
    for mmx in range(-nspread, nspread): #mm index for all the spreading points
        for mmy in range(-nspread,nspread):
            for mmz in range(-nspread,nspread):
                #griding with g(x,y) = exp(-(x^2 + y^2) / 4*tau)
                #ftau[(mx + mmx) % nf1, (my + mmy) % nf2, (mz + mmz) % nf3] +=
                tmp = c[i] * exp(-0.25 * (\
                (xi - hx * (mx + mmx)) ** 2 + \
                (yi - hy * (my + mmy)) ** 2 + \
                (zi - hz * (mz + mmz)) ** 2 ) / tau)
                cuda.atomic.add(real_ftau, ((mx + mmx) % nf1, (my + mmy) % nf2, (mz + mmz) % nf3), tmp.real)
                cuda.atomic.add(imag_ftau, ((mx + mmx) % nf1, (my + mmy) % nf2, (mz + mmz) % nf3), tmp.imag)

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
    real_ftau = np.zeros(ftau.shape, dtype=np.float64)
    imag_ftau = np.zeros(ftau.shape, dtype=np.float64) #atom add only support float32 or 64
    gaussker_3d1_cuda[bpg, tpb](x, y, z, c, hx, hy, hz, nf1, nf2, nf3, nspread, tau, real_ftau, imag_ftau)
    ftau = real_ftau + 1j*imag_ftau
    return ftau


#type1, fast version with precompute of exponentials
@cuda.jit
def gaussker_3d1_fast_cuda(x, y, z, c, hx, hy, hz, nf1, nf2, nf3, nspread, tau, E3, real_ftau, imag_ftau ):
    """This kernel function for gauss grid 1d typ1, and it will be executed by a thread."""
    i     = cuda.grid(1)
    if i > x.shape[0]:
        return
    #read x, y, z values
    xi    = x[i] % (2 * np.pi) #x, shift the source point xj so that it lies in [0,2*pi]
    yi    = y[i] % (2 * np.pi) #y, shift the source point yj so that it lies in [0,2*pi]
    zi    = z[i] % (2 * np.pi) #z, shift the source point zj so that it lies in [0,2*pi]
    mx    = 1 + int(xi // hx) #index for the closest grid point
    my    = 1 + int(yi // hy) #index for the closest grid point
    mz    = 1 + int(zi // hz) #index for the closest grid point
    xi    = (xi - hx * mx) #offsets from the closest grid point
    yi    = (yi - hy * my) #offsets from the closest grid point
    zi    = (zi - hz * mz) #offsets from the closest grid point
    # precompute E1, E2x, E2y, E2z
    E1    = exp(-0.25 * (xi ** 2 + yi ** 2 + zi ** 2) / tau)
    E2x   = exp((xi * np.pi) / (nf1 * tau))
    E2y   = exp((yi * np.pi) / (nf2 * tau))
    E2z   = exp((zi * np.pi) / (nf3 * tau))
    V0    = c[i] * E1
    #do the 3d griding here,
    #use the symmetry of E1, E2 and E3, e.g. 1/(E2mmz*E2z) = 1/(E2x**(mmx)*E2x) = E2x**(-mmx-1)
    E2mmx = 1#update with E2mmx *= E2x <-> E2mmx = E2x**(mmz) in the middle loop
    for mmx in range(nspread):#mm index for all the spreading points
        E2mmy = 1#update with E2mmy *= E2y <-> E2mmy = E2y**(mmy) in the middle loop
        for mmy in range(nspread):#mm index for all the spreading points
            E2mmz = 1#update with E2mmz *= E2z <-> E2mmz = E2z**(mmz) in the middle loop
            for mmz in range(nspread):#mm index for all the spreading points
                #ftau[    (mx + mmx) % nf1,     (my + mmy) % nf2,     (mz + mmz) % nf3] +=
                tmpxpypzp = V0 * E2mmx       * E2mmy       * E2mmz        * E3[    mmx,     mmy,     mmz]
                #ftau[    (mx + mmx) % nf1, (my - mmy - 1) % nf2,     (mz + mmz) % nf3] +=
                tmpxpynzp = V0 * E2mmx       / (E2mmy*E2y) * E2mmz        * E3[    mmx, mmy + 1,     mmz]
                #ftau[(mx - mmx - 1) % nf1,     (my + mmy) % nf2,     (mz + mmz) % nf3] +=
                tmpxnypzp = V0 / (E2mmx*E2x) * E2mmy       * E2mmz        * E3[mmx + 1,     mmy,     mmz]
                #ftau[(mx - mmx - 1) % nf1, (my - mmy - 1) % nf2,     (mz + mmz) % nf3] +=
                tmpxnynzp = V0 / (E2mmx*E2x) / (E2mmy*E2y) * E2mmz        * E3[mmx + 1, mmy + 1,     mmz]
                #ftau[    (mx + mmx) % nf1,     (my + mmy) % nf2, (mz - mmz - 1) % nf3] +=
                tmpxpypzn = V0 * E2mmx       * E2mmy       / (E2mmz*E2z)  * E3[    mmx,     mmy, mmz + 1]
                #ftau[    (mx + mmx) % nf1, (my - mmy - 1) % nf2, (mz - mmz - 1) % nf3] +=
                tmpxpynzn = V0 * E2mmx       / (E2mmy*E2y) / (E2mmz*E2z)  * E3[    mmx, mmy + 1, mmz + 1]
                #ftau[(mx - mmx - 1) % nf1,     (my + mmy) % nf2, (mz - mmz - 1) % nf3] +=
                tmpxnypzn = V0 / (E2mmx*E2x) * E2mmy       / (E2mmz*E2z)  * E3[mmx + 1,     mmy, mmz + 1]
                #ftau[(mx - mmx - 1) % nf1, (my - mmy - 1) % nf2, (mz - mmz - 1) % nf3] +=
                tmpxnynzn = V0 / (E2mmx*E2x) / (E2mmy*E2y) / (E2mmz*E2z)  * E3[mmx + 1, mmy + 1, mmz + 1]
                #use atom sum here
                cuda.atomic.add(real_ftau, (    (mx + mmx) % nf1,     (my + mmy) % nf2,     (mz + mmz) % nf3), tmpxpypzp.real) #x  1, y  1, z  1
                cuda.atomic.add(imag_ftau, (    (mx + mmx) % nf1,     (my + mmy) % nf2,     (mz + mmz) % nf3), tmpxpypzp.imag)
                cuda.atomic.add(real_ftau, (    (mx + mmx) % nf1, (my - mmy - 1) % nf2,     (mz + mmz) % nf3), tmpxpynzp.real) #x  1, y -1, z  1
                cuda.atomic.add(imag_ftau, (    (mx + mmx) % nf1, (my - mmy - 1) % nf2,     (mz + mmz) % nf3), tmpxpynzp.imag)
                cuda.atomic.add(real_ftau, ((mx - mmx - 1) % nf1,     (my + mmy) % nf2,     (mz + mmz) % nf3), tmpxnypzp.real) #x -1, y  1, z  1
                cuda.atomic.add(imag_ftau, ((mx - mmx - 1) % nf1,     (my + mmy) % nf2,     (mz + mmz) % nf3), tmpxnypzp.imag)
                cuda.atomic.add(real_ftau, ((mx - mmx - 1) % nf1, (my - mmy - 1) % nf2,     (mz + mmz) % nf3), tmpxnynzp.real) #x -1, y -1, z  1
                cuda.atomic.add(imag_ftau, ((mx - mmx - 1) % nf1, (my - mmy - 1) % nf2,     (mz + mmz) % nf3), tmpxnynzp.imag)
                cuda.atomic.add(real_ftau, (    (mx + mmx) % nf1,     (my + mmy) % nf2, (mz - mmz - 1) % nf3), tmpxpypzn.real) #x  1, y  1, z  1
                cuda.atomic.add(imag_ftau, (    (mx + mmx) % nf1,     (my + mmy) % nf2, (mz - mmz - 1) % nf3), tmpxpypzn.imag)
                cuda.atomic.add(real_ftau, (    (mx + mmx) % nf1, (my - mmy - 1) % nf2, (mz - mmz - 1) % nf3), tmpxpynzn.real) #x  1, y -1, z  1
                cuda.atomic.add(imag_ftau, (    (mx + mmx) % nf1, (my - mmy - 1) % nf2, (mz - mmz - 1) % nf3), tmpxpynzn.imag)
                cuda.atomic.add(real_ftau, ((mx - mmx - 1) % nf1,     (my + mmy) % nf2, (mz - mmz - 1) % nf3), tmpxnypzn.real) #x -1, y  1, z  1
                cuda.atomic.add(imag_ftau, ((mx - mmx - 1) % nf1,     (my + mmy) % nf2, (mz - mmz - 1) % nf3), tmpxnypzn.imag)
                cuda.atomic.add(real_ftau, ((mx - mmx - 1) % nf1, (my - mmy - 1) % nf2, (mz - mmz - 1) % nf3), tmpxnynzn.real) #x -1, y -1, z  1
                cuda.atomic.add(imag_ftau, ((mx - mmx - 1) % nf1, (my - mmy - 1) % nf2, (mz - mmz - 1) % nf3), tmpxnynzn.imag)
                E2mmz *= E2z
            E2mmy *= E2y
        E2mmx *= E2x

def build_grid_3d1_fast_cuda( x, y, z, c, tau, nspread, ftau, E3 ):
    #number of pioints along x, y, z
    nf1 = ftau.shape[0]
    nf2 = ftau.shape[1]
    nf3 = ftau.shape[2]
    #minimal intervals along x, y, z
    hx = 2 * np.pi / nf1
    hy = 2 * np.pi / nf2
    hz = 2 * np.pi / nf3
    # precompute E3 exponential
    for p in range(nspread + 1):
        for l in range(nspread + 1):
            for j in range(nspread + 1):
                E3[j,l,p] = exp(-((np.pi * j / nf1) ** 2 + (np.pi * l / nf2) ** 2 + (np.pi * p / nf3) ** 2)/ tau)
    #prepare for CUDA, compute CUDA parameters n, tpb, bpg
    device    = cuda.get_current_device()
    n         = x.shape[0] #number of kernels in the computing
    tpb       = device.WARP_SIZE
    bpg       = int(np.ceil(float(n)/tpb))
    real_ftau = np.zeros(ftau.shape,np.float64)
    imag_ftau = np.zeros(ftau.shape,np.float64)
    #computing start here
    gaussker_3d1_fast_cuda[bpg, tpb](x, y, z, c, hx, hy, hz, nf1, nf2, nf3, nspread, tau, E3, real_ftau, imag_ftau )
    ftau = real_ftau + 1j*imag_ftau
    return ftau

#type 2
@cuda.jit
def gaussker_3d2_cuda(x, y, z, c, hx, hy, hz, nf1, nf2, nf3, nspread, tau, fntau ):
    """This kernel function for gauss grid 3d type2, and it will be executed by a thread."""
    i  = cuda.grid(1)
    if i > x.shape[0]:
        return

    xi = x[i] % (2.0 * np.pi) #x, shift the source point xj so that it lies in [0,2*pi]
    yi = y[i] % (2.0 * np.pi) #y, shift the source point yj so that it lies in [0,2*pi]
    zi = z[i] % (2.0 * np.pi) #z, shift the source point zj so that it lies in [0,2*pi]
    mx = 1 + int(xi // hx) #index for the closest grid point
    my = 1 + int(yi // hy) #index for the closest grid point
    mz = 1 + int(zi // hz) #index for the closest grid point
    #do the 3d griding here
    for mmx in range(-nspread, nspread): #mm index for all the spreading points
        for mmy in range(-nspread,nspread):#mm index for all the spreading points
            for mmz in range(-nspread,nspread):#mm index for all the spreading points
                #griding with g(x,y) = exp(-(x^2 + y^2 + z^2) / 4*tau)
                c[i] \
                += fntau[(mx + mmx) % nf1, (my + mmy) % nf2, (mz + mmz) % nf3] * exp(-0.25 * (\
                (xi - hx * (mx + mmx)) ** 2 + \
                (yi - hy * (my + mmy)) ** 2 + \
                (zi - hz * (mz + mmz)) ** 2 ) / tau)

def build_grid_3d2_cuda( x, y, z, fntau, tau, nspread ):
    #number of pioints along x, y, z
    nf1 = fntau.shape[0]
    nf2 = fntau.shape[1]
    nf3 = fntau.shape[2]
    #minimal intervals along x, y, z
    hx = 2 * np.pi / nf1
    hy = 2 * np.pi / nf2
    hz = 2 * np.pi / nf3
    #c is coefficients
    c = np.zeros(x.shape,dtype = fntau.dtype)
    #prepare for CUDA, compute CUDA parameters n, tpb, bpg
    device = cuda.get_current_device()
    n = x.shape[0] #number of kernels in the computing
    tpb = device.WARP_SIZE
    bpg = int(np.ceil(float(n)/tpb))
    #computing start here
    gaussker_3d2_cuda[bpg, tpb](x, y, z, c, hx, hy, hz, nf1, nf2, nf3, nspread, tau, fntau)
    #type2 nufft has 1/N term
    return c/(nf1*nf2*nf3)

#type 2 3d, fast version with precomputing of exponentials
@cuda.jit
def gaussker_3d2_fast_cuda(x, y, z, c, hx, hy, hz, nf1, nf2, nf3, nspread, tau, E3, fntau ):
    """This kernel function for gauss grid 3d type2 with precomputing of exponentials, and it will be executed by a thread."""
    i  = cuda.grid(1)
    if i > x.shape[0]:
        return
    #read x, y, z values
    xi = x[i] % (2.0 * np.pi) #x, shift the source point xj so that it lies in [0,2*pi]
    yi = y[i] % (2.0 * np.pi) #y, shift the source point yj so that it lies in [0,2*pi]
    zi = z[i] % (2.0 * np.pi) #z, shift the source point zj so that it lies in [0,2*pi]
    #index for the closest grid point
    mx = 1 + int(xi // hx) #index for the closest grid point
    my = 1 + int(yi // hy) #index for the closest grid point
    mz = 1 + int(zi // hz) #index for the closest grid point
    #offsets from the closest grid point
    xi = (xi - hx * mx) #
    yi = (yi - hy * my)
    zi = (zi - hz * mz)
    # precompute E1, E2x, E2y, E2z
    E1 = exp(-0.25 * (xi ** 2 + yi ** 2 + zi ** 2) / tau)
    E2x = exp((xi * np.pi) / (nf1 * tau))
    E2y = exp((yi * np.pi) / (nf2 * tau))
    E2z = exp((zi * np.pi) / (nf3 * tau))
    #do the 3d griding here,
    #use the symmetry of E1, E2 and E3, e.g. 1/(E2mmx*E2x)=1/(E2x**(mmx)*E2x) = E2x**(-mmx-1)
    E2mmx = 1 #update with E2mmx *= E2x <-> E2mmx=E2x**(mmx) in the outer loop
    for mmx in range(nspread):#mm index for all the spreading points
        E2mmy = 1 #update with E2mmy *= E2y <-> E2mmy=E2y**(mmy) in the middle loop
        for mmy in range(nspread):#mm index for all the spreading points
            E2mmz = 1 #update with E2mmz *= E2z <-> E2mmz=E2z**(mmz) in the middle loop
            for mmz in range(nspread):#mm index for all the spreading points
                #griding with g(x,y) = exp(-(x^2 + y^2 + z^2) / 4*tau) = E1 * E2mmx * E2mmy * E2mmz * E3
                c[i] += fntau[    (mx + mmx) % nf1,     (my + mmy) % nf2,     (mz + mmz) % nf3] * E1 * E2mmx       * E2mmy       * E2mmz        * E3[    mmx,     mmy,     mmz]
                c[i] += fntau[    (mx + mmx) % nf1, (my - mmy - 1) % nf2,     (mz + mmz) % nf3] * E1 * E2mmx       / (E2mmy*E2y) * E2mmz        * E3[    mmx, mmy + 1,     mmz]
                c[i] += fntau[(mx - mmx - 1) % nf1,     (my + mmy) % nf2,     (mz + mmz) % nf3] * E1 / (E2mmx*E2x) * E2mmy       * E2mmz        * E3[mmx + 1,     mmy,     mmz]
                c[i] += fntau[(mx - mmx - 1) % nf1, (my - mmy - 1) % nf2,     (mz + mmz) % nf3] * E1 / (E2mmx*E2x) / (E2mmy*E2y) * E2mmz        * E3[mmx + 1, mmy + 1,     mmz]
                c[i] += fntau[    (mx + mmx) % nf1,     (my + mmy) % nf2, (mz - mmz - 1) % nf3] * E1 * E2mmx       * E2mmy       / (E2mmz*E2z)  * E3[    mmx,     mmy, mmz + 1]
                c[i] += fntau[    (mx + mmx) % nf1, (my - mmy - 1) % nf2, (mz - mmz - 1) % nf3] * E1 * E2mmx       / (E2mmy*E2y) / (E2mmz*E2z)  * E3[    mmx, mmy + 1, mmz + 1]
                c[i] += fntau[(mx - mmx - 1) % nf1,     (my + mmy) % nf2, (mz - mmz - 1) % nf3] * E1 / (E2mmx*E2x) * E2mmy       / (E2mmz*E2z)  * E3[mmx + 1,     mmy, mmz + 1]
                c[i] += fntau[(mx - mmx - 1) % nf1, (my - mmy - 1) % nf2, (mz - mmz - 1) % nf3] * E1 / (E2mmx*E2x) / (E2mmy*E2y) / (E2mmz*E2z)  * E3[mmx + 1, mmy + 1, mmz + 1]
                E2mmz *= E2z
            E2mmy *= E2y
        E2mmx *= E2x

def build_grid_3d2_fast_cuda( x, y, z, fntau, tau, nspread, E3 ):
    #number of pioints along x, y, z
    nf1 = fntau.shape[0]
    nf2 = fntau.shape[1]
    nf3 = fntau.shape[2]
    #minimal intervals along x, y, z
    hx = 2 * np.pi / nf1
    hy = 2 * np.pi / nf2
    hz = 2 * np.pi / nf3
    #c is coefficients
    c = np.zeros(x.shape,dtype = fntau.dtype)
    # precompute some exponentials
    for p in range(nspread + 1):
        for l in range(nspread + 1):
            for j in range(nspread + 1):
                E3[j,l,p] = exp(-((np.pi * j / nf1) ** 2 + (np.pi * l / nf2) ** 2 + (np.pi * p / nf3) ** 2)/ tau)
    #prepare for CUDA, compute CUDA parameters n, tpb, bpg
    device = cuda.get_current_device()
    n = x.shape[0] #number of kernels in the computing
    tpb = device.WARP_SIZE
    bpg = int(np.ceil(float(n)/tpb))
    #compute_E3[bpg, tpb](nspread, nf1, nf2, nf3, tau, E3)
    #computing start here
    gaussker_3d2_fast_cuda[bpg, tpb](x, y, z, c, hx, hy, hz, nf1, nf2, nf3, nspread, tau, E3, fntau)
    #type2 nufft has 1/N term
    return c/(nf1*nf2*nf3)

#type 2 & type 1
#input fntau, output real/imag_ftau
@cuda.jit
def gaussker_3d21_cuda(x, y, z, hx, hy, hz, nf1, nf2, nf3, nspread, tau, fntau, real_ftau, imag_ftau ):
    """This kernel function for gauss grid 3d type2, and it will be executed by a thread."""
    i  = cuda.grid(1)
    if i > x.shape[0]:
        return

    xi = x[i] % (2.0 * np.pi) #x, shift the source point xj so that it lies in [0,2*pi]
    yi = y[i] % (2.0 * np.pi) #y, shift the source point yj so that it lies in [0,2*pi]
    zi = z[i] % (2.0 * np.pi) #z, shift the source point zj so that it lies in [0,2*pi]
    mx = 1 + int(xi // hx) #index for the closest grid point
    my = 1 + int(yi // hy) #index for the closest grid point
    mz = 1 + int(zi // hz) #index for the closest grid point
    c  = 0.0 #coefficient, saved temporarily
    #do the 3d griding here
    for mmx in range(-nspread, nspread): #mm index for all the spreading points
        for mmy in range(-nspread,nspread):#mm index for all the spreading points
            for mmz in range(-nspread,nspread):#mm index for all the spreading points
                #griding with g(x,y) = exp(-(x^2 + y^2 + z^2) / 4*tau)
                c += fntau[(mx + mmx) % nf1, (my + mmy) % nf2, (mz + mmz) % nf3] * exp(-0.25 * (\
                (xi - hx * (mx + mmx)) ** 2 + \
                (yi - hy * (my + mmy)) ** 2 + \
                (zi - hz * (mz + mmz)) ** 2 ) / tau)
    #type2 nufft has 1/N term
    c = c/(nf1*nf2*nf3) 
    #grid again
    for mmx in range(-nspread, nspread): #mm index for all the spreading points
        for mmy in range(-nspread,nspread):
            for mmz in range(-nspread,nspread):
                #griding with g(x,y) = exp(-(x^2 + y^2) / 4*tau)
                #ftau[(mx + mmx) % nf1, (my + mmy) % nf2, (mz + mmz) % nf3] +=
                tmp = c * exp(-0.25 * (\
                (xi - hx * (mx + mmx)) ** 2 + \
                (yi - hy * (my + mmy)) ** 2 + \
                (zi - hz * (mz + mmz)) ** 2 ) / tau)
                cuda.atomic.add(real_ftau, ((mx + mmx) % nf1, (my + mmy) % nf2, (mz + mmz) % nf3), tmp.real)
                cuda.atomic.add(imag_ftau, ((mx + mmx) % nf1, (my + mmy) % nf2, (mz + mmz) % nf3), tmp.imag)

def build_grid_3d21_cuda( x, y, z, ftau, tau, nspread ):
    #number of pioints along x, y, z
    nf1 = ftau.shape[0]
    nf2 = ftau.shape[1]
    nf3 = ftau.shape[2]
    #minimal intervals along x, y, z
    hx = 2 * np.pi / nf1
    hy = 2 * np.pi / nf2
    hz = 2 * np.pi / nf3
    #c is coefficients
    #c = np.zeros(x.shape,dtype = ftau.dtype)
    #prepare for CUDA, compute CUDA parameters n, tpb, bpg
    device = cuda.get_current_device()
    n = x.shape[0] #number of kernels in the computing
    tpb = device.WARP_SIZE
    bpg = int(np.ceil(float(n)/tpb))
    real_ftau = np.zeros(ftau.shape, dtype=np.float64)
    imag_ftau = np.zeros(ftau.shape, dtype=np.float64) #atom add only support float32 or 64
    #computing start here
    gaussker_3d21_cuda[bpg, tpb](x, y, z, hx, hy, hz, nf1, nf2, nf3, nspread, tau, ftau, real_ftau, imag_ftau)
    ftau = real_ftau + 1j*imag_ftau
    return ftau



"""
mian function of nufft3d
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
#3d nufft type 1
def nufft3d1_gaussker_cuda( x, y, z, c, ms, mt, mu, df=1.0, eps=1E-15, iflag=1, gridfast=1 ):
    """Fast Non-Uniform Fourier Transform with Numba"""
    nspread, nf1, nf2, nf3, tau = nufft_func._compute_3d_grid_params(ms, mt, mu, eps)
    #try to override nspread
    #nspread = min(3, nspread)

    # Construct the convolved grid
    if gridfast is 0:
        Ftau = build_grid_3d1_cuda(x * df, y * df, z *df, c, tau, nspread,\
                      np.zeros((nf1, nf2, nf3), dtype=c.dtype))
    else:#precompute some exponentials, not working
        Ftau = build_grid_3d1_fast_cuda(x * df, y * df, z *df, c, tau, nspread,\
                      np.zeros((nf1, nf2, nf3), dtype=c.dtype), \
                      np.zeros((nspread+1, nspread+1, nspread+1), dtype=c.dtype))

    # Compute the FFT on the convolved grid
    if iflag < 0:
        Ftau = (1 / (nf1 * nf2 * nf3)) * fftnc2c_cuda(Ftau)#np.fft.fftn(Ftau,s=None,axes=(0,1,2))#
    else:
        Ftau = ifftnc2c_cuda(Ftau)#np.fft.ifftn(Ftau,s=None,axes=(0,1,2))#

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
def nufft3d2_gaussker_cuda( x, y, z, Fk, ms, mt, mu, df=1.0, eps=1E-15, iflag=1, gridfast=0 ):
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
        Fntau = (nf1 * nf2 * nf3) * ifftnc2c_cuda(Fntau)#np.fft.ifftn(Fntau,s=None,axes=(0,1,2))#
    else:
        Fntau = fftnc2c_cuda(Fntau)#np.fft.fftn(Fntau,s=None,axes=(0,1,2))#

    # Construct the convolved grid
    if gridfast is not 1:
        fx = build_grid_3d2_cuda(x * df, y * df, z * df, Fntau, tau, nspread)
    else:
        fx = build_grid_3d2_fast_cuda(x * df, y * df, z * df, Fntau, tau, nspread,\
         np.zeros((nspread+1, nspread+1, nspread+1), dtype=Fk.dtype))
    return fx

#3d unfft type 2 & type 1
def nufft3d21_gaussker_cuda( x, y, z, Fk, ms, mt, mu, df=1.0, eps=1E-15, iflag=1, gridfast=0 ):
    """Fast Non-Uniform Fourier Transform with Numba"""
    nspread, nf1, nf2, nf3, tau = nufft_func._compute_3d_grid_params(ms, mt, mu, eps)

    # Deconvolve the grid using convolution theorem, Ftau * G(k1)^-1
    k1, k2, k3 = nufft_func.nufftfreqs3d(ms, mt, mu)
    # Note the np.sqrt(np.pi / tau)**3 due to the 3 dimentions of nufft
    Fk = np.sqrt(np.pi / tau)**3 * np.exp(tau * (k1 ** 2 + k2 ** 2 + k3 ** 2)) * Fk

    #reshape to match the size Ftau 
    Ftau = np.zeros((nf1, nf2, nf3), dtype=Fk.dtype)
    Ftau[-(ms//2):      ,       -(mt//2):,       -(mu//2):] = Fk[0:ms//2 , 0:mt//2 , 0:mu//2]# 1 1 1
    Ftau[:ms//2 + ms % 2,       -(mt//2):,       -(mu//2):] = Fk[ms//2:ms, 0:mt//2 , 0:mu//2]# 2 1 1
    Ftau[-(ms//2):      ,       -(mt//2):, :mu//2 + mu % 2] = Fk[0:ms//2 , 0:mt//2 ,mu//2:mu]# 1 1 2
    Ftau[-(ms//2):      , :mt//2 + mt % 2,       -(mu//2):] = Fk[0:ms//2 ,mt//2:mt , 0:mu//2]# 1 2 1
    Ftau[:ms//2 + ms % 2, :mt//2 + mt % 2,       -(mu//2):] = Fk[ms//2:ms,mt//2:mt , 0:mu//2]# 2 2 1
    Ftau[:ms//2 + ms % 2, :mt//2 + mt % 2, :mu//2 + mu % 2] = Fk[ms//2:ms,mt//2:mt ,mu//2:mu]# 2 2 2
    Ftau[:ms//2 + ms % 2,       -(mt//2):, :mu//2 + mu % 2] = Fk[ms//2:ms, 0:mt//2 ,mu//2:mu]# 2 1 2
    Ftau[-(ms//2):      , :mt//2 + mt % 2, :mu//2 + mu % 2] = Fk[0:ms//2 ,mt//2:mt ,mu//2:mu]# 1 2 2

    # Compute the FFT on the convolved grid
    if iflag < 0:
        Ftau = (nf1 * nf2 * nf3) * ifftnc2c_cuda(Ftau)#np.fft.ifftn(Ftau,s=None,axes=(0,1,2))#
    else:
        Ftau = fftnc2c_cuda(Ftau)#np.fft.fftn(Ftau,s=None,axes=(0,1,2))#

    # Construct the convolved grid
    if 1 :#gridfast is not 1:
        Ftau = build_grid_3d21_cuda(x * df, y * df, z * df, Ftau, tau, nspread)
    #else:
    #    Ftau = build_grid_3d21_fast_cuda(x * df, y * df, z * df, Ftau, tau, nspread,\
    #     np.zeros((nspread+1, nspread+1, nspread+1), dtype=Fk.dtype))

    # Compute the FFT on the convolved grid
    if iflag < 0:
        Ftau = (1 / (nf1 * nf2 * nf3)) * fftnc2c_cuda(Ftau)#np.fft.fftn(Ftau,s=None,axes=(0,1,2))#
    else:
        Ftau = ifftnc2c_cuda(Ftau)#np.fft.ifftn(Ftau,s=None,axes=(0,1,2))#

    #ut.plotim3(np.absolute(Ftau[:,:,:]))
    #truncate the Ftau to match the size of output, alias are removed
    Ftau = np.concatenate([Ftau[-(ms//2):,:,:], Ftau[:ms//2 + ms % 2,:,:]],0)
    Ftau = np.concatenate([Ftau[:,-(mt//2):,:], Ftau[:,:mt//2 + mt % 2,:]],1)
    Ftau = np.concatenate([Ftau[:,:,-(mu//2):], Ftau[:,:,:mu//2 + mu % 2]],2)
    # Deconvolve the grid using convolution theorem, Ftau * G(k1,k2,k3)^-1
    # Note the np.sqrt(np.pi / tau)**3 due to the 3 dimentions of nufft
    return (1 / len(x)) * np.sqrt(np.pi / tau)**3 * \
    np.exp(tau * (k1 ** 2 + k2 ** 2 + k3 ** 2)) * Ftau


def test():
    #test nufft type1
    #nufft_test_func.time_nufft1d1(nufft1d1_gaussker_cuda,64,5120,5)
    #nufft_test_func.time_nufft2d1(nufft2d1_gaussker_cuda,64,64,5120)
    #nufft_test_func.time_nufft3d1(nufft3d1_gaussker_cuda,256,256,128,20480,1)

    #test nufft type2
    #nufft_test_func.time_nufft1d2(nufft1d1_gaussker_cuda,nufft1d2_gaussker_cuda,32,102400,5)
    #nufft_test_func.time_nufft2d2(nufft2d1_gaussker_cuda,nufft2d2_gaussker_cuda,32,32,5120,5)
    #nufft_test_func.time_nufft3d2(nufft3d1_gaussker_cuda,nufft3d2_gaussker_cuda,128,128,128,20480,1)

    #compare
    #nufft_test_func.compare_nufft1d1(nufft_func.nudft1d1, nufft1d1_gaussker_cuda,32,32000)
    #nufft_test_func.compare_nufft2d1(nufft_func.nudft2d1, nufft2d1_gaussker_cuda, 64, 64,5120)
    #nufft_test_func.compare_nufft3d1(nufft_func.nudft3d1, nufft3d1_gaussker_cuda, 16, 16,16,20480)

    #compare type 2& typ1
    nufft_test_func.compare_nufft1d21(nufft1d1_gaussker_cuda, nufft1d21_gaussker_cuda, 128, 100000,1)
    nufft_test_func.compare_nufft2d21(nufft2d1_gaussker_cuda, nufft2d21_gaussker_cuda,16,16,25000,1)
    nufft_test_func.compare_nufft3d21(nufft3d1_gaussker_cuda, nufft3d21_gaussker_cuda,18,16,16,204800,1)    

if __name__ == "__main__":
    test()
