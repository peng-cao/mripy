from __future__ import print_function, division
import numpy as np
import numba
import utilities.utilities_func as ut
import matplotlib.pyplot as plt
import nufft_test_func
from numba import cuda
from time import time
from math import exp

#####################################################################################################
#ft direct calculation, 1d,2d,3d, type1
#####################################################################################################

"""
this is code modified based on examples from
https://jakevdp.github.io/blog/2015/02/24/optimizing-python-with-numpy-and-numba/
this is direct computation of nonuniform FFT 1d type 1

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
the direct ft, output dim is ms X 1

"""
#nudft direct calculation
def nudft1d1( x, c, ms, df=1.0, iflag=1 ):
    """Non-Uniform Direct Fourier Transform"""
    sign = -1 if iflag < 0 else 1
    return (1 / len(x)) * np.dot(c, np.exp(sign * 1j * nufftfreqs1d(ms, df) * x[:, np.newaxis]))

"""
#create frequence ranges:
k1 range = -ms/2,...,ms/2;
inputs:
ms is length of k1 range
outputs:
k1 shape is ms X 1
"""
def nufftfreqs1d( ms, df=1 ):
    """Compute the frequency range used in nufft for ms frequency bins"""
    return df * np.arange(-(ms // 2), ms - (ms // 2))

"""
this is direct computation of nonuniform FFT 2d type 1

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
the direct ft, output dim is ms X mt

"""
#nudft direct calculation
def nudft2d1( x , y, c, ms, mt, df=1.0, iflag=1 ):
    """Non-Uniform Direct Fourier Transform"""
    sign = -1 if iflag < 0 else 1
    k1, k2 = nufftfreqs2d(ms, mt, df)
    x = x.transpose() #x dim should be N X 1
    y = y.transpose() #y dim should be N X 1
    return (1 / len(x)) * np.dot(c, np.exp(sign * 1j * (\
        k1 * x[:, np.newaxis, np.newaxis] + \
        k2 * y[:, np.newaxis, np.newaxis]).transpose([1,0,2])))


"""
#create frequence ranges:
k1 range = -ms/2,...,ms/2;
k2 range = -mt/2,...,mt/2;
inputs:
ms is length of k1 range
mt is length of k2 range
outputs:
k1 shape is ms X mt X mu, it increase along the first dim
k2 shape is ms X mt X mu, it increase along the second dim
"""
def nufftfreqs2d( ms, mt, df=1.0 ):
    """Compute the frequency range used in nufft for ms frequency bins"""
    return df * np.mgrid[-(ms // 2): ms - (ms // 2), -(mt // 2): mt - (mt // 2)]


"""
this is direct computation of nonuniform FFT 3d type 1

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
iflag determine whether -1 or 1 sign for exp(+/- i k1 x),
exp(+/- i k2 y) and exp(+/- i k3 z); default is 1

output:
the direct ft, output dim is ms X mt X mu

"""
#nudft direct calculation
def nudft3d1( x, y, z, c, ms, mt, mu, df=1.0, iflag=1 ):
    """Non-Uniform Direct Fourier Transform"""
    sign = -1 if iflag < 0 else 1
    k1, k2, k3 = nufftfreqs3d(ms, mt, mu, df)
    x = x.transpose() #x dim should be N X 1
    y = y.transpose() #y dim should be N X 1
    z = z.transpose() #z dim should be N X 1
    return (1 / len(x)) * np.dot(c, np.exp(sign * 1j * (\
        k1 * x[:, np.newaxis, np.newaxis, np.newaxis] + \
        k2 * y[:, np.newaxis, np.newaxis, np.newaxis] + \
        k3 * z[:, np.newaxis, np.newaxis, np.newaxis]).transpose([1, 2, 0, 3])))

"""
#create frequence ranges:
k1 range = -ms/2,...,ms/2;
k2 range = -mt/2,...,mt/2;
k3 range = -mu/2,...,mu/2;
inputs:
ms is length of k1 range
mt is length of k2 range
mu is length of k3 range
outputs:
k1 shape is ms X mt X mu, it increase along the first dim
k2 shape is ms X mt X mu, it increase along the second dim
k3 shape is ms X mt X mu, it increase along the third dim
"""
def nufftfreqs3d( ms, mt, mu, df=1.0 ):
    """Compute the frequency range used in nufft for ms frequency bins"""
    return df * np.mgrid[\
    -(ms // 2): ms - (ms // 2),\
    -(mt // 2): mt - (mt // 2),\
    -(mu // 2): mu - (mu // 2)]


"""
this is code modified based on examples on
https://jakevdp.github.io/blog/2015/02/24/optimizing-python-with-numpy-and-numba/
the method is describled in
Accelerating the Nonuniform Fast Fourier Transform by Greengard, Leslie and Lee, June-yub
some descriptions are from nufft1d1f90()
References:
[DR] Fast Fourier transforms for nonequispaced data,
     A. Dutt and V. Rokhlin, SIAM J. Sci. Comput. 14,
     1368-1383, 1993.

[GL] Accelerating the Nonuniform Fast Fourier Transform,
     L. Greengard and J.-Y. Lee, SIAM Review 46, 443-454 (2004).

The oversampled regular mesh is defined by
nf1 = rat*ms  points, where rat is the oversampling rat.

For simplicity, we set
    rat = 2 for eps > 1.0d-11
    rat = 3 for eps <= 1.0d-11.

It can be shown [DR] that the precision eps is achieved when
nspread = int(-log(eps)/(pi*(rat-1d0)/(rat-.5d0)) + .5d0)
and tau is chosen as
     tau = pi*lambda/(ms**2)
     lambda = nspread/(rat(rat-0.5)).
"""

def _compute_1d_grid_params( ms, eps ):
    # Choose Nspread & tau from eps following Dutt & Rokhlin (1993)
    if eps <= 1E-33 or eps >= 1E-1:
        raise ValueError("eps = {0:.0e}; must satisfy "
                         "1e-33 < eps < 1e-1.".format(eps))
    rat = 2 if eps > 1E-11 else 3
    nspread = int(-np.log(eps) / (np.pi * (rat - 1) / (rat - 0.5)) + 0.5)
    nf1 = max(rat * ms, 2 * nspread)
    lambda_ = nspread / (rat * (rat - 0.5))
    tau = np.pi * lambda_ / ms ** 2
    return nspread, nf1, tau

"""
The oversampled regular mesh is defined by
nf1 = rat*ms  points, where rat is the oversampling ratio.
nf2 = rat*mt  points, where rat is the oversampling ratio.
"""
def _compute_2d_grid_params( ms, mt, eps ):
    # Choose Nspread & tau from eps following Dutt & Rokhlin (1993)
    if eps <= 1E-33 or eps >= 1E-1:
        raise ValueError("eps = {0:.0e}; must satisfy "
                         "1e-33 < eps < 1e-1.".format(eps))
    rat = 2 if eps > 1E-11 else 3
    nspread = int(-np.log(eps) / (np.pi * (rat - 1) / (rat - 0.5)) + 0.5)
    nf1 = max(rat * ms, 2 * nspread)
    nf2 = max(rat * mt, 2 * nspread)
    lambda_ = nspread / (rat * (rat - 0.5))
    tau = np.pi * lambda_ / ms ** 2
    return nspread, nf1, nf2, tau

"""
The oversampled regular mesh is defined by
nf1 = rat*ms  points, where rat is the oversampling ratio.
nf2 = rat*mt  points, where rat is the oversampling ratio.
nf3 = rat*mu  points, where rat is the oversampling ratio.
"""
def _compute_3d_grid_params( ms, mt, mu, eps ):
    # Choose Nspread & tau from eps following Dutt & Rokhlin (1993)
    if eps <= 1E-33 or eps >= 1E-1:
        raise ValueError("eps = {0:.0e}; must satisfy "
                         "1e-33 < eps < 1e-1.".format(eps))
    rat = 2 if eps > 1E-11 else 3
    nspread = int(-np.log(eps) / (np.pi * (rat - 1) / (rat - 0.5)) + 0.5)
    nf1 = max(rat * ms, 2 * nspread)
    nf2 = max(rat * mt, 2 * nspread)
    nf3 = max(rat * mu, 2 * nspread)
    lambda_ = nspread / (rat * (rat - 0.5))
    tau =  np.pi * lambda_ / ms ** 2
    return nspread, nf1, nf2, nf3, tau


#####################################################################################################
#unfft 1d, type1, type2 and type2&type1 (AHA)
#####################################################################################################

"""
this is code modified based on examples on
https://jakevdp.github.io/blog/2015/02/24/optimizing-python-with-numpy-and-numba/
the method is describled in
Accelerating the Nonuniform Fast Fourier Transform by Greengard, Leslie and Lee, June-yub
Some descriptions are from nufft1d1f90()
Fast Gaussian gridding is based on the following observation.
#calculate the grid, type 1
# nopython=True, nogil=True means an error will be raised if fast compilation is not possible.
@numba.jit(nopython=True, nogil=True)
def build_grid_1d1( x, c, tau, nspread, ftau ):
    nf1 = ftau.shape[0]
    hx = 2 * np.pi / nf1
    for i in range(x.shape[0]):
        xi = x[i] % (2 * np.pi) #x, shift the source point xj so that it lies in [0,2*pi]
        m = 1 + int(xi // hx) #index for the closest grid point
        for mm in range(-nspread, nspread): #mm index for all the spreading points
            ftau[(m + mm) % nf1] += c[i] * np.exp(-0.25 * (xi - hx * (m + mm)) ** 2 / tau)
            #griding with g(x) = exp(-x^2 / 4*tau)
    return ftau

Let hx = 2*pi/nf1. In gridding data onto a regular mesh with
spacing nf1, we shift the source point xj so that it lies in [0,2*pi]
to simplify the calculations. Since we are viewing the function as periodic,
this has no effect on the result.

The Gaussian used for convolution is:
g(x) = exp(-x^2 / 4*tau)

"""
#calculate the grid, type 1
# nopython=True, nogil=True means an error will be raised if fast compilation is not possible.
@numba.jit(nopython=True, nogil=True)
def build_grid_1d1( x, c, tau, nspread, ftau ):
    nf1 = ftau.shape[0]
    hx = 2 * np.pi / nf1
    for i in range(x.shape[0]):
        xi = x[i] % (2 * np.pi) #x, shift the source point xj so that it lies in [0,2*pi]
        m = 1 + int(xi // hx) #index for the closest grid point
        for mm in range(-nspread, nspread): #mm index for all the spreading points
            ftau[(m + mm) % nf1] += c[i] * np.exp(-0.25 * (xi - hx * (m + mm)) ** 2 / tau)
            #griding with g(x) = exp(-x^2 / 4*tau)
    return ftau



#type1, fast version with precompute of exponentials
@numba.jit(nopython=True, nogil=True)
def build_grid_1d1_fast( x, c, tau, nspread, ftau, E3 ):
    nf1 = ftau.shape[0]
    hx = 2 * np.pi / nf1
    # precompute some exponents
    for j in range(nspread + 1):
        E3[j] = np.exp(-(np.pi * j / nf1) ** 2 / tau)

    # spread values onto ftau
    for i in range(x.shape[0]):
        xi = x[i] % (2 * np.pi) #x
        m = 1 + int(xi // hx) #index for the closest grid point
        xi = (xi - hx * m) #
        E1 = np.exp(-0.25 * xi ** 2 / tau)
        E2 = np.exp((xi * np.pi) / (nf1 * tau))
        E2mm = 1
        for mm in range(nspread):
            ftau[(m + mm) % nf1] += c[i] * E1 * E2mm * E3[mm]
            E2mm *= E2
            ftau[(m - mm - 1) % nf1] += c[i] * E1 / E2mm * E3[mm + 1]
    return ftau

#1d grid type 2
@numba.jit(nopython=True, nogil=True)
def build_grid_1d2( x, fntau, tau, nspread ):
    nf1 = fntau.shape[0]
    hx = 2 * np.pi / nf1
    c = np.zeros(x.shape,dtype=fntau.dtype)
    for i in range(x.shape[0]):
        xi = x[i] % (2 * np.pi) #x, shift the source point xj so that it lies in [0,2*pi]
        m = 1 + int(xi // hx) #index for the closest grid point
        for mm in range(-nspread, nspread): #mm index for all the spreading points
            c[i] += fntau[(m + mm) % nf1] * np.exp(-0.25 * (xi - hx * (m + mm)) ** 2 / tau)#griding with g(x) = exp(-x^2 / 4*tau)
    return c/nf1

#1d grid type 2 fast
@numba.jit(nopython=True, nogil=True)
def build_grid_1d2_fast( x, fntau, tau, nspread, E3 ):
    nf1 = fntau.shape[0]
    hx = 2 * np.pi / nf1
    c = np.zeros(x.shape,dtype=fntau.dtype)
    # precompute some exponents
    for j in range(nspread + 1):
        E3[j] = np.exp(-(np.pi * j / nf1) ** 2 / tau)
    # spread values onto fntau or c
    for i in range(x.shape[0]):
        xi = x[i] % (2 * np.pi) #x
        m = 1 + int(xi // hx) #index for the closest grid point
        xi = (xi - hx * m) #
        E1 = np.exp(-0.25 * xi ** 2 / tau)
        E2 = np.exp((xi * np.pi) / (nf1 * tau))
        E2mm = 1
        for mm in range(nspread):
            c[i] += fntau[(m + mm) % nf1] * E1 * E2mm * E3[mm]
            E2mm *= E2
            c[i] += fntau[(m - mm - 1) % nf1] * E1 / E2mm * E3[mm + 1]
    return c/nf1

#1d grid type 2 & type 1
@numba.jit(nopython=True, nogil=True)
def build_grid_1d21( x, fntau, tau, nspread ):
    # had problem when ftau is used for both input and output, current version seperate it as fntau/ftau
    nf1  = fntau.shape[0]
    hx   = 2 * np.pi / nf1
    Em   = np.zeros(2*nspread + 1, dtype=fntau.dtype)#will reuse this exponential
    ftau = np.zeros(fntau.shape,   dtype=fntau.dtype)
    for i in range(x.shape[0]):
        c  = 0.0 #coefficient, saved temporarily
        xi = x[i] % (2 * np.pi) #x, shift the source point xj so that it lies in [0,2*pi]
        m  = 1 + int(xi // hx) #index for the closest grid point
        #grid once
        for mm in range(-nspread, nspread): #mm index for all the spreading points
            Em[mm + nspread] = np.exp(-0.25 * (xi - hx * (m + mm)) ** 2 / tau)
            c  += fntau[(m + mm) % nf1] * Em[mm + nspread]#griding with g(x) = exp(-x^2 / 4*tau)
        #grid again
        c  = c/nf1
        for mm in range(-nspread, nspread): #mm index for all the spreading points
            ftau[(m + mm) % nf1]  += c * Em[mm + nspread]
            #griding with g(x) = exp(-x^2 / 4*tau)
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
def nufft1d1_gaussker( x, c, ms, df=1.0, eps=1E-15, iflag=1, gridfast=1 ):
    """Fast Non-Uniform Fourier Transform with Numba"""
    nspread, nf1, tau = _compute_1d_grid_params(ms, eps)

    #compute Ftaushape
    if len(c.shape) > 1:
        Ftaushape = (nf1) + c.shape[1:len(c.shape)]
    else:
        Ftaushape = (nf1)


    if gridfast is 0:
        # Construct the convolved grid
        Ftau = build_grid_1d1(x * df, c, tau, nspread, np.zeros(Ftaushape, dtype=c.dtype))
    else:#fast griding with precomputing of some expoentials
        Ftau = build_grid_1d1_fast(x * df, c, tau, nspread, np.zeros(Ftaushape, dtype=c.dtype),\
                           np.zeros(nspread + 1, dtype=c.dtype))

    # Compute the FFT on the convolved grid
    if iflag < 0:
        Ftau = (1 / nf1) * np.fft.fft(Ftau, axis = 0)
    else:
        Ftau = np.fft.ifft(Ftau, axis = 0)
    #truncate the Ftau to match the size of output, alias are removed
    Ftau = np.concatenate([Ftau[-(ms//2):], Ftau[:ms//2 + ms % 2]])
    # Deconvolve the grid using convolution theorem, Ftau * G(k1)^-1
    k1 = nufftfreqs1d(ms)
    #match dimensions
    outkshape, outFkshape = ut.dim_match(k1.shape ,Ftau.shape)
    return (1 / len(x)) * np.sqrt(np.pi / tau) \
         * np.multiply(np.exp(tau * k1 ** 2).reshape(outkshape), Ftau.reshape(outFkshape))

#1d nufft type 2
def nufft1d2_gaussker( x, Fk, ms, df=1.0, eps=1E-15, iflag=1, gridfast=1 ):
    """Fast Non-Uniform Fourier Transform with Numba"""
    nspread, nf1, tau = _compute_1d_grid_params(ms, eps)
    # Deconvolve the grid using convolution theorem, Ftau * G(k1)^-1
    k1 = nufftfreqs1d(ms)
    #match dimensions
    outkshape, outFkshape = ut.dim_match(k1.shape ,Fk.shape)
    Fk = np.sqrt(np.pi / tau) \
        * np.multiply(np.exp(tau * k1 ** 2).reshape(outkshape), Fk.reshape(outFkshape))

    #compute Ftau shape
    if len(Fk.shape) > 1:
        Ftaushape = (nf1,) + Fk.shape[1:len(Fk.shape)]
    else:
        Ftaushape = nf1
    #reshape Fk and fftshift to match the size Fntau or Ftau
    Fntau = np.zeros(Ftaushape, dtype=Fk.dtype)
    Fntau[-(ms//2):] = Fk[0:ms//2]
    Fntau[:ms//2 + ms % 2] = Fk[ms//2:ms]

    # Compute the FFT on the convolved grid
    if iflag < 0:
        Fntau = nf1 * np.fft.ifft(Fntau, axis = 0)
    else:
        Fntau = np.fft.fft(Fntau, axis = 0)

    # Construct the convolved grid
    if gridfast is not 1:
        fx = build_grid_1d2(x*df, Fntau, tau, nspread)
    else:
        fx = build_grid_1d2_fast(x*df, Fntau, tau, nspread, np.zeros(nspread + 1, dtype=Fk.dtype))
    return fx

#1d nufft type 2 & type 1
def nufft1d21_gaussker( x, Fk, ms, df=1.0, eps=1E-15, iflag=1, gridfast=0 ):
    """Fast Non-Uniform Fourier Transform with Numba"""
    nspread, nf1, tau = _compute_1d_grid_params(ms, eps)

    # Deconvolve the grid using convolution theorem, Ftau * G(k1)^-1
    k1 = nufftfreqs1d(ms)
    #match dimensions
    outkshape, outFkshape = ut.dim_match(k1.shape ,Fk.shape)
    Fk = np.sqrt(np.pi / tau) \
        * np.multiply(np.exp(tau * k1 ** 2).reshape(outkshape), Fk.reshape(outFkshape))

    #compute Ftau shape
    if len(Fk.shape) > 1:
        Ftaushape = (nf1,) + Fk.shape[1:len(Fk.shape)]
    else:
        Ftaushape = nf1
    #reshape Fk and fftshift to match the size Fntau or Ftau
    Ftau = np.zeros(Ftaushape, dtype=Fk.dtype)
    Ftau[-(ms//2):] = Fk[0:ms//2]
    Ftau[:ms//2 + ms % 2] = Fk[ms//2:ms]

    # Compute the FFT on the convolved grid
    if iflag < 0:
        Ftau = nf1 * np.fft.ifft(Ftau, axis = 0)
    else:
        Ftau = np.fft.fft(Ftau, axis = 0)

    # Construct the convolved grid
    if 1:#gridfast is not 1:
        Ftau = build_grid_1d21(x*df, Ftau, tau, nspread)
    #else:
        #Ftau = build_grid_1d21_fast(x/df, Ftau, tau, nspread, np.zeros(nspread + 1, dtype=Fk.dtype))

    # Compute the FFT on the convolved grid
    if iflag < 0:
        Ftau = (1 / nf1) * np.fft.fft(Ftau)
    else:
        Ftau = np.fft.ifft(Ftau)
    #truncate the Ftau to match the size of output, alias are removed
    Ftau = np.concatenate([Ftau[-(ms//2):], Ftau[:ms//2 + ms % 2]])
    return (1 / len(x)) * np.sqrt(np.pi / tau) \
         * np.multiply(np.exp(tau * k1 ** 2).reshape(outkshape), Ftau.reshape(outFkshape))

#####################################################################################################
#unfft 2d, type1, type2 and type2&type1 (AHA)
#####################################################################################################

#2d grid type 1
@numba.jit(nopython=True, nogil=True)
def build_grid_2d1( x, y, c, tau, nspread, ftau ):
    nf1 = ftau.shape[0]
    nf2 = ftau.shape[1]
    hx = 2 * np.pi / nf1
    hy = 2 * np.pi / nf2
    for i in range(x.shape[0]):
        xi = x[i] % (2 * np.pi) #x, shift the source point xj so that it lies in [0,2*pi]
        yi = y[i] % (2 * np.pi) #y, shift the source point yj so that it lies in [0,2*pi]
        m1 = 1 + int(xi // hx) #index for the closest grid point
        m2 = 1 + int(yi // hy) #index for the closest grid point
        for mm1 in range(-nspread, nspread): #mm index for all the spreading points
            for mm2 in range(-nspread,nspread):
                #griding with g(x,y) = exp(-(x^2 + y^2) / 4*tau)
                ftau[(m1 + mm1) % nf1, (m2 + mm2) % nf2] \
                += c[i] * np.exp(-0.25 * (\
                (xi - hx * (m1 + mm1)) ** 2 + \
                (yi - hy * (m2 + mm2)) ** 2 ) / tau)
    return ftau

#type1 2d grid, fast version with precompute of exponentials
@numba.jit(nopython=True, nogil=True)
def build_grid_2d1_fast( x, y, c, tau, nspread, ftau, E3 ):
    nf1 = ftau.shape[0]
    nf2 = ftau.shape[1]
    hx = 2 * np.pi / nf1
    hy = 2 * np.pi / nf2
    # precompute some exponents
    for l in range(nspread + 1):
        for j in range(nspread + 1):
            E3[j,l] = np.exp(-((np.pi * j / nf1) ** 2 + (np.pi * l /nf2) ** 2)/ tau)
    # spread values onto ftau
    for i in range(x.shape[0]):
        xi = x[i] % (2 * np.pi) #x
        yi = y[i] % (2 * np.pi) #y
        mx = 1 + int(xi // hx) #index for the closest grid point
        my = 1 + int(yi // hy)
        xi = (xi - hx * mx) #
        yi = (yi - hy * my)
        E1 = np.exp(-0.25 * (xi ** 2 + yi ** 2) / tau)
        E2x = np.exp((xi * np.pi) / (nf1 * tau))
        E2y = np.exp((yi * np.pi) / (nf2 * tau))
        E2mmx = 1
        V0 = c[i] * E1
        for mmx in range(nspread):
            E2mmy = 1
            for mmy in range(nspread):#use the symmetry of E1, E2 and E3, e.g. 1/(E2(mmx)*E2x) = E2(mx-mmx)
                ftau[    (mx + mmx) % nf1,     (my + mmy) % nf2] += V0 * E2mmx       * E2mmy       * E3[    mmx,     mmy]
                ftau[    (mx + mmx) % nf1, (my - mmy - 1) % nf2] += V0 * E2mmx       / (E2mmy*E2y) * E3[    mmx, mmy + 1]
                ftau[(mx - mmx - 1) % nf1,     (my + mmy) % nf2] += V0 / (E2mmx*E2x) * E2mmy       * E3[mmx + 1,     mmy]
                ftau[(mx - mmx - 1) % nf1, (my - mmy - 1) % nf2] += V0 / (E2mmx*E2x) / (E2mmy*E2y) * E3[mmx + 1, mmy + 1]
                E2mmy *= E2y
            E2mmx *= E2x
    return ftau

#2d type 2
@numba.jit(nopython=True, nogil=True)
def build_grid_2d2( x, y, fntau, tau, nspread ):
    nf1 = fntau.shape[0]
    nf2 = fntau.shape[1]
    hx = 2 * np.pi / nf1
    hy = 2 * np.pi / nf2
    c = np.zeros(x.shape,dtype = fntau.dtype)
    for i in range(x.shape[0]):
        xi = x[i] % (2 * np.pi) #x, shift the source point xj so that it lies in [0,2*pi]
        yi = y[i] % (2 * np.pi) #y, shift the source point yj so that it lies in [0,2*pi]
        m1 = 1 + int(xi // hx) #index for the closest grid point
        m2 = 1 + int(yi // hy) #index for the closest grid point
        for mm1 in range(-nspread, nspread): #mm index for all the spreading points
            for mm2 in range(-nspread,nspread):
                #griding with g(x,y) = exp(-(x^2 + y^2) / 4*tau)
                c[i] \
                += fntau[(m1 + mm1) % nf1, (m2 + mm2) % nf2] * np.exp(-0.25 * (\
                (xi - hx * (m1 + mm1)) ** 2 + \
                (yi - hy * (m2 + mm2)) ** 2 ) / tau)
    return c/(nf1*nf2)

#type2 2d grid, fast version with precompute of exponentials
@numba.jit(nopython=True, nogil=True)
def build_grid_2d2_fast( x, y, fntau, tau, nspread, E3 ):
    nf1 = fntau.shape[0]
    nf2 = fntau.shape[1]
    hx = 2 * np.pi / nf1
    hy = 2 * np.pi / nf2
    c = np.zeros(x.shape,dtype = fntau.dtype)
    # precompute some exponents
    for l in range(nspread + 1):
        for j in range(nspread + 1):
            E3[j,l] = np.exp(-((np.pi * j / nf1) ** 2 + (np.pi * l /nf2) ** 2)/ tau)
    # spread values onto ftau
    for i in range(x.shape[0]):
        xi = x[i] % (2 * np.pi) #x
        yi = y[i] % (2 * np.pi) #y
        mx = 1 + int(xi // hx) #index for the closest grid point
        my = 1 + int(yi // hy)
        xi = (xi - hx * mx) #
        yi = (yi - hy * my)
        E1 = np.exp(-0.25 * (xi ** 2 + yi ** 2) / tau)
        E2x = np.exp((xi * np.pi) / (nf1 * tau))
        E2y = np.exp((yi * np.pi) / (nf2 * tau))
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
    return c/(nf1*nf2)

#2d type 2 & type 1
@numba.jit(nopython=True, nogil=True)
def build_grid_2d21( x, y, fntau, tau, nspread ):
    nf1  = fntau.shape[0]
    nf2  = fntau.shape[1]
    hx   = 2 * np.pi / nf1
    hy   = 2 * np.pi / nf2
    Em   = np.zeros((2*nspread+1, 2*nspread+1), dtype = fntau.dtype)#will reuse this exponential
    ftau = np.zeros(fntau.shape, dtype = fntau.dtype)
    for i in range(x.shape[0]):
        c  = 0.0 #coefficient, saved temporarily
        xi = x[i] % (2 * np.pi) #x, shift the source point xj so that it lies in [0,2*pi]
        yi = y[i] % (2 * np.pi) #y, shift the source point yj so that it lies in [0,2*pi]
        m1 = 1 + int(xi // hx) #index for the closest grid point
        m2 = 1 + int(yi // hy) #index for the closest grid point
        #grid once
        for mm1 in range(-nspread, nspread): #mm index for all the spreading points
            for mm2 in range(-nspread,nspread):
                #griding with g(x,y) = exp(-(x^2 + y^2) / 4*tau)
                Em[mm1 + nspread, mm2 + nspread] =  np.exp(-0.25 * (\
                (xi - hx * (m1 + mm1)) ** 2 + \
                (yi - hy * (m2 + mm2)) ** 2 ) / tau)
                c += fntau[(m1 + mm1) % nf1, (m2 + mm2) % nf2] \
                     * Em[mm1 + nspread, mm2 + nspread]
        #grid again
        c = c/(nf1*nf2)
        for mm1 in range(-nspread, nspread): #mm index for all the spreading points
            for mm2 in range(-nspread,nspread):
                #griding with g(x,y) = exp(-(x^2 + y^2) / 4*tau)
                ftau[(m1 + mm1) % nf1, (m2 + mm2) % nf2] += \
                      c * Em[mm1 + nspread, mm2 + nspread]
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
#2d nufft type 1
def nufft2d1_gaussker( x, y, c, ms, mt, df=1.0, eps=1E-15, iflag=1, gridfast=1 ):
    """Fast Non-Uniform Fourier Transform with Numba"""
    nspread, nf1, nf2, tau = _compute_2d_grid_params(ms, mt, eps)

    #compute Ftaushape
    if len(c.shape) > 1:
        Ftaushape = (nf1, nf2) + c.shape[1:len(c.shape)]
    else:
        Ftaushape = (nf1, nf2)

    # Construct the convolved grid
    #Ftau = build_grid_2d1(x * df, y * df, c, tau, nspread,
    #                  np.zeros((nf1, nf2), dtype = c.dtype))
    if gridfast is 0:
        Ftau = build_grid_2d1(x * df, y * df, c, tau, nspread, np.zeros(Ftaushape, dtype=c.dtype))
    else:#griding with precomputing of some exponentials
        Ftau = build_grid_2d1_fast(x * df, y * df, c, tau, nspread, np.zeros(Ftaushape, dtype=c.dtype),\
                           np.zeros((nspread + 1, nspread + 1), dtype=c.dtype))

    # Compute the FFT on the convolved grid
    if iflag < 0:
        Ftau = (1 / (nf1 * nf2)) * np.fft.fft2(Ftau, axes = (0,1))
    else:
        Ftau = np.fft.ifft2(Ftau, axes = (0,1)) #1/(nf1*nf2) in ifft2 function when norm = None

    #truncate the Ftau to match the size of output, alias are removed
    Ftau = np.concatenate([Ftau[-(ms//2):,:], Ftau[:ms//2 + ms % 2,:]],0)
    Ftau = np.concatenate([Ftau[:,-(mt//2):], Ftau[:,:mt//2 + mt % 2]],1)

    # Deconvolve the grid using convolution theorem, Ftau * G(k1,k2)^-1
    k1,k2 = nufftfreqs2d(ms, mt)

    #match dimensions
    outkshape, outFkshape = ut.dim_match(k1.shape ,Ftau.shape)
    # Note the np.sqrt(np.pi / tau)**2 due to the 2 dimentions of nufft
    return (1 / len(x)) * np.sqrt(np.pi / tau)**2 \
        * np.multiply(np.exp(tau * (k1 ** 2 + k2 ** 2)).reshape(outkshape), Ftau.reshape(outFkshape)) #

#2d nufft type 2
def nufft2d2_gaussker( x, y, Fk, ms, mt, df=1.0, eps=1E-15, iflag=1, gridfast=1 ):
    """Fast Non-Uniform Fourier Transform with Numba"""
    nspread, nf1, nf2, tau = _compute_2d_grid_params(ms, mt, eps)

    # Deconvolve the grid using convolution theorem, Ftau * G(k1)^-1
    k1, k2 = nufftfreqs2d(ms, mt)

    #match dimensions
    outkshape, outFkshape = ut.dim_match(k1.shape ,Fk.shape)
    # Note the np.sqrt(np.pi / tau)**2 due to the 2 dimentions of nufft
    Fk = np.sqrt(np.pi / tau)**2 \
       * np.multiply(np.exp(tau * (k1 ** 2 + k2 ** 2)).reshape(outkshape), Fk.reshape(outFkshape)) #np.sqrt(np.pi / tau) *

    #compute Ftau shape
    if len(Fk.shape) > 2:
        Ftaushape = (nf1, nf2) + Fk.shape[2:len(Fk.shape)]
    else:
        Ftaushape = (nf1, nf2)
    #reshape Fk and fftshift to match the size Fntau or Ftau
    Fntau = np.zeros(Ftaushape, dtype=Fk.dtype)
    Fntau[ -(ms//2):       ,       -(mt//2): ] = Fk[ 0:ms//2  ,  0:mt//2 ]#1 1
    Fntau[ :ms//2 + ms % 2 , :mt//2 + mt % 2 ] = Fk[ ms//2:ms , mt//2:mt ]#2 2
    Fntau[ :ms//2 + ms % 2 ,       -(mt//2): ] = Fk[ ms//2:ms ,  0:mt//2 ]#2 1
    Fntau[ -(ms//2):       , :mt//2 + mt % 2 ] = Fk[ 0:ms//2  , mt//2:mt ]#1 2

    # Compute the FFT on the convolved grid
    if iflag < 0:
        Fntau = nf1 * nf2 * np.fft.ifft2(Fntau, axes = (0,1))
    else:
        Fntau = np.fft.fft2(Fntau, axes = (0,1))
    # Construct the convolved grid
    if gridfast is not 1:
        fx = build_grid_2d2(x*df, y*df, Fntau, tau, nspread)
    else:
        fx = build_grid_2d2_fast(x*df, y*df, Fntau, tau, nspread,\
            np.zeros((nspread + 1, nspread + 1), dtype=Fk.dtype))
    return fx

#2d nufft type 2 & type 1
def nufft2d21_gaussker( x, y, Fk, ms, mt, df=1.0, eps=1E-15, iflag=1, gridfast=1 ):
    """Fast Non-Uniform Fourier Transform with Numba"""
    nspread, nf1, nf2, tau = _compute_2d_grid_params(ms, mt, eps)

    # Deconvolve the grid using convolution theorem, Ftau * G(k1)^-1
    k1, k2 = nufftfreqs2d(ms, mt)

    #match dimensions
    outkshape, outFkshape = ut.dim_match(k1.shape ,Fk.shape)
    # Note the np.sqrt(np.pi / tau)**2 due to the 2 dimentions of nufft
    Fk = np.sqrt(np.pi / tau)**2 \
       * np.multiply(np.exp(tau * (k1 ** 2 + k2 ** 2)).reshape(outkshape), Fk.reshape(outFkshape)) #np.sqrt(np.pi / tau) *

    #compute Ftau shape
    if len(Fk.shape) > 2:
        Ftaushape = (nf1, nf2) + Fk.shape[2:len(Fk.shape)]
    else:
        Ftaushape = (nf1, nf2)
    #reshape Fk and fftshift to match the size Fntau or Ftau
    Ftau = np.zeros(Ftaushape, dtype=Fk.dtype)
    Ftau[ -(ms//2):       ,       -(mt//2): ] = Fk[ 0:ms//2  ,  0:mt//2 ]#1 1
    Ftau[ :ms//2 + ms % 2 , :mt//2 + mt % 2 ] = Fk[ ms//2:ms , mt//2:mt ]#2 2
    Ftau[ :ms//2 + ms % 2 ,       -(mt//2): ] = Fk[ ms//2:ms ,  0:mt//2 ]#2 1
    Ftau[ -(ms//2):       , :mt//2 + mt % 2 ] = Fk[ 0:ms//2  , mt//2:mt ]#1 2

    # Compute the FFT on the convolved grid
    if iflag < 0:
        Ftau = nf1 * nf2 * np.fft.ifft2(Ftau, axes = (0,1))
    else:
        Ftau = np.fft.fft2(Ftau, axes = (0,1))
    # Construct the convolved grid
    if 1:#gridfast is not 1:
        Ftau = build_grid_2d21(x*df, y*df, Ftau, tau, nspread)
    #else:
    #    Ftau = build_grid_2d21_fast(x/df, y/df, Ftau, tau, nspread,\
    #        np.zeros((nspread + 1, nspread + 1), dtype=Fk.dtype))

    # Compute the FFT on the convolved grid
    if iflag < 0:
        Ftau = (1 / (nf1 * nf2)) * np.fft.fft2(Ftau, axes = (0,1))
    else:
        Ftau = np.fft.ifft2(Ftau, axes = (0,1)) #1/(nf1*nf2) in ifft2 function when norm = None

    #truncate the Ftau to match the size of output, alias are removed
    Ftau = np.concatenate([Ftau[-(ms//2):,:], Ftau[:ms//2 + ms % 2,:]],0)
    Ftau = np.concatenate([Ftau[:,-(mt//2):], Ftau[:,:mt//2 + mt % 2]],1)

    # Deconvolve the grid using convolution theorem, Ftau * G(k1,k2)^-1
    # Note the np.sqrt(np.pi / tau)**2 due to the 2 dimentions of nufft
    return (1 / len(x)) * np.sqrt(np.pi / tau)**2 \
        * np.multiply(np.exp(tau * (k1 ** 2 + k2 ** 2)).reshape(outkshape), Ftau.reshape(outFkshape)) #
#####################################################################################################
#unfft 3d, type1, type2 and type2&type1 (AHA)
#####################################################################################################

#3d grid type 1
@numba.jit(nopython=True, nogil=True,parallel=True)
def build_grid_3d1( x, y, z, c, tau, nspread, ftau ):
    nf1 = ftau.shape[0]
    nf2 = ftau.shape[1]
    nf3 = ftau.shape[2]
    hx = 2 * np.pi / nf1
    hy = 2 * np.pi / nf2
    hz = 2 * np.pi / nf3
    for i in range(x.shape[0]):
        xi = x[i] % (2 * np.pi) #x, shift the source point xj so that it lies in [0,2*pi]
        yi = y[i] % (2 * np.pi) #y, shift the source point yj so that it lies in [0,2*pi]
        zi = z[i] % (2 * np.pi) #z, shift the source point zj so that it lies in [0,2*pi]
        m1 = 1 + int(xi // hx) #index for the closest grid point
        m2 = 1 + int(yi // hy) #index for the closest grid point
        m3 = 1 + int(zi // hz) #index for the closest grid point
        for mm1 in range(-nspread, nspread): #mm index for all the spreading points
            for mm2 in range(-nspread,nspread):
                for mm3 in range(-nspread,nspread):
                    #griding with g(x,y) = exp(-(x^2 + y^2) / 4*tau)
                    ftau[(m1 + mm1) % nf1, (m2 + mm2) % nf2, (m3 + mm3) % nf3] \
                    += c[i] * np.exp(-0.25 * (\
                    (xi - hx * (m1 + mm1)) ** 2 + \
                    (yi - hy * (m2 + mm2)) ** 2 + \
                    (zi - hz * (m3 + mm3)) ** 2 ) / tau)
    return ftau

#type1 3d grid, fast version with precompute of exponentials
@numba.jit(nopython=True,nogil=True)#, parallel=True
def build_grid_3d1_fast( x, y, z, c, tau, nspread, ftau, E3 ):
    nf1 = ftau.shape[0]
    nf2 = ftau.shape[1]
    nf3 = ftau.shape[2]
    hx = 2 * np.pi / nf1
    hy = 2 * np.pi / nf2
    hz = 2 * np.pi / nf3
    # precompute some exponents
    for p in range(nspread + 1):
        for l in range(nspread + 1):
            for j in range(nspread + 1):
                E3[j,l,p] = np.exp(-((np.pi * j / nf1) ** 2 + (np.pi * l / nf2) ** 2 + (np.pi * p / nf3) ** 2)/ tau)
    # spread values onto ftau
    for i in range(x.shape[0]):
        xi = x[i] % (2 * np.pi) #x
        yi = y[i] % (2 * np.pi) #y
        zi = z[i] % (2 * np.pi)
        mx = 1 + int(xi // hx) #index for the closest grid point
        my = 1 + int(yi // hy)
        mz = 1 + int(zi // hz)
        xi = (xi - hx * mx) #
        yi = (yi - hy * my)
        zi = (zi - hz * mz)
        E1 = np.exp(-0.25 * (xi ** 2 + yi ** 2 + zi ** 2) / tau)
        E2x = np.exp((xi * np.pi) / (nf1 * tau))
        E2y = np.exp((yi * np.pi) / (nf2 * tau))
        E2z = np.exp((zi * np.pi) / (nf3 * tau))
        V0 = c[i] * E1
        E2mmx = 1
        for mmx in range(nspread):
            E2mmy = 1
            for mmy in range(nspread):#use the symmetry of E1, E2 and E3, e.g. 1/(E2(mmx)*E2x) = E2(mx-mmx)
                E2mmz = 1
                for mmz in range(nspread):
                    ftau[    (mx + mmx) % nf1,     (my + mmy) % nf2,     (mz + mmz) % nf3] += V0 * E2mmx       * E2mmy       * E2mmz        * E3[    mmx,     mmy,     mmz]
                    ftau[    (mx + mmx) % nf1, (my - mmy - 1) % nf2,     (mz + mmz) % nf3] += V0 * E2mmx       / (E2mmy*E2y) * E2mmz        * E3[    mmx, mmy + 1,     mmz]
                    ftau[(mx - mmx - 1) % nf1,     (my + mmy) % nf2,     (mz + mmz) % nf3] += V0 / (E2mmx*E2x) * E2mmy       * E2mmz        * E3[mmx + 1,     mmy,     mmz]
                    ftau[(mx - mmx - 1) % nf1, (my - mmy - 1) % nf2,     (mz + mmz) % nf3] += V0 / (E2mmx*E2x) / (E2mmy*E2y) * E2mmz        * E3[mmx + 1, mmy + 1,     mmz]
                    ftau[    (mx + mmx) % nf1,     (my + mmy) % nf2, (mz - mmz - 1) % nf3] += V0 * E2mmx       * E2mmy       / (E2mmz*E2z)  * E3[    mmx,     mmy, mmz + 1]
                    ftau[    (mx + mmx) % nf1, (my - mmy - 1) % nf2, (mz - mmz - 1) % nf3] += V0 * E2mmx       / (E2mmy*E2y) / (E2mmz*E2z)  * E3[    mmx, mmy + 1, mmz + 1]
                    ftau[(mx - mmx - 1) % nf1,     (my + mmy) % nf2, (mz - mmz - 1) % nf3] += V0 / (E2mmx*E2x) * E2mmy       / (E2mmz*E2z)  * E3[mmx + 1,     mmy, mmz + 1]
                    ftau[(mx - mmx - 1) % nf1, (my - mmy - 1) % nf2, (mz - mmz - 1) % nf3] += V0 / (E2mmx*E2x) / (E2mmy*E2y) / (E2mmz*E2z)  * E3[mmx + 1, mmy + 1, mmz + 1]
                    E2mmz *= E2z
                E2mmy *= E2y
            E2mmx *= E2x
    return ftau

#3d grid type 2
@numba.jit(nopython=True, nogil=True)
def build_grid_3d2( x, y, z, fntau, tau, nspread ):
    nf1 = fntau.shape[0]
    nf2 = fntau.shape[1]
    nf3 = fntau.shape[2]
    hx = 2 * np.pi / nf1
    hy = 2 * np.pi / nf2
    hz = 2 * np.pi / nf3
    c = np.zeros(x.shape,dtype = fntau.dtype)
    for i in range(x.shape[0]):
        xi = x[i] % (2 * np.pi) #x, shift the source point xj so that it lies in [0,2*pi]
        yi = y[i] % (2 * np.pi) #y, shift the source point yj so that it lies in [0,2*pi]
        zi = z[i] % (2 * np.pi) #z, shift the source point zj so that it lies in [0,2*pi]
        m1 = 1 + int(xi // hx) #index for the closest grid point
        m2 = 1 + int(yi // hy) #index for the closest grid point
        m3 = 1 + int(zi // hz) #index for the closest grid point
        for mm1 in range(-nspread, nspread): #mm index for all the spreading points
            for mm2 in range(-nspread,nspread):
                for mm3 in range(-nspread,nspread):
                    #griding with g(x,y) = exp(-(x^2 + y^2) / 4*tau)
                    c[i] += fntau[(m1 + mm1) % nf1, (m2 + mm2) % nf2, (m3 + mm3) % nf3] * \
                     np.exp(-0.25 * (\
                    (xi - hx * (m1 + mm1)) ** 2 + \
                    (yi - hy * (m2 + mm2)) ** 2 + \
                    (zi - hz * (m3 + mm3)) ** 2 ) / tau)
    return c/(nf1*nf2*nf3)

#type 2 3d grid, fast version with precompute of exponentials
@numba.jit(nopython=True, nogil=True)
def build_grid_3d2_fast( x, y, z, fntau, tau, nspread, E3 ):
    nf1 = fntau.shape[0]
    nf2 = fntau.shape[1]
    nf3 = fntau.shape[2]
    hx = 2 * np.pi / nf1
    hy = 2 * np.pi / nf2
    hz = 2 * np.pi / nf3
    c = np.zeros(x.shape,dtype = fntau.dtype)
    # precompute some exponents
    for p in range(nspread + 1):
        for l in range(nspread + 1):
            for j in range(nspread + 1):
                E3[j,l,p] = np.exp(-((np.pi * j / nf1) ** 2 + (np.pi * l / nf2) ** 2 + (np.pi * p / nf3) ** 2)/ tau)
    # spread values onto ftau
    for i in range(x.shape[0]):
        xi = x[i] % (2 * np.pi) #x
        yi = y[i] % (2 * np.pi) #y
        zi = z[i] % (2 * np.pi)
        mx = 1 + int(xi // hx) #index for the closest grid point
        my = 1 + int(yi // hy)
        mz = 1 + int(zi // hz)
        xi = (xi - hx * mx) #
        yi = (yi - hy * my)
        zi = (zi - hz * mz)
        E1 = np.exp(-0.25 * (xi ** 2 + yi ** 2 + zi ** 2) / tau) #missed zi ** 2 find@20170415
        E2x = np.exp((xi * np.pi) / (nf1 * tau))
        E2y = np.exp((yi * np.pi) / (nf2 * tau))
        E2z = np.exp((zi * np.pi) / (nf3 * tau))

        E2mmx = 1
        for mmx in range(nspread):
            E2mmy = 1
            for mmy in range(nspread):#use the symmetry of E1, E2 and E3, e.g. 1/(E2(mmx)*E2x) = E2(mx-mmx)
                E2mmz = 1
                for mmz in range(nspread):
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
    return c/(nf1*nf2*nf3)

#3d grid type 2 & type 1
@numba.jit(nopython=True, nogil=True)
def build_grid_3d21( x, y, z, fntau, tau, nspread ):
    nf1   = fntau.shape[0]
    nf2   = fntau.shape[1]
    nf3   = fntau.shape[2]
    hx    = 2 * np.pi / nf1
    hy    = 2 * np.pi / nf2
    hz    = 2 * np.pi / nf3
    Em    = np.zeros((2*nspread+1,2*nspread+1,2*nspread+1),dtype = fntau.dtype) #will reuse this exponential
    ftau  = np.zeros(fntau.shape, dtype = fntau.dtype)
    for i in range(x.shape[0]):
        c  = 0.0 #coefficient, saved temporarily
        xi = x[i] % (2 * np.pi) #x, shift the source point xj so that it lies in [0,2*pi]
        yi = y[i] % (2 * np.pi) #y, shift the source point yj so that it lies in [0,2*pi]
        zi = z[i] % (2 * np.pi) #z, shift the source point zj so that it lies in [0,2*pi]
        m1 = 1 + int(xi // hx) #index for the closest grid point
        m2 = 1 + int(yi // hy) #index for the closest grid point
        m3 = 1 + int(zi // hz) #index for the closest grid point
        #grid once
        for mm1 in range(-nspread, nspread): #mm index for all the spreading points
            for mm2 in range(-nspread,nspread):
                for mm3 in range(-nspread,nspread):
                    #griding with g(x,y) = exp(-(x^2 + y^2) / 4*tau)
                    Em[mm1 + nspread, mm2 + nspread, mm3 + nspread] = \
                    np.exp(-0.25 * (\
                    (xi - hx * (m1 + mm1)) ** 2 + \
                    (yi - hy * (m2 + mm2)) ** 2 + \
                    (zi - hz * (m3 + mm3)) ** 2 ) / tau)
                    c += fntau[(m1 + mm1) % nf1, (m2 + mm2) % nf2, (m3 + mm3) % nf3]\
                         * Em[mm1 + nspread, mm2 + nspread, mm3 + nspread]
        #grid again
        c = c/(nf1*nf2*nf3)
        for mm1 in range(-nspread, nspread): #mm index for all the spreading points
            for mm2 in range(-nspread,nspread):
                for mm3 in range(-nspread,nspread):
                    #griding with g(x,y) = exp(-(x^2 + y^2 + z^2) / 4*tau)
                    ftau[(m1 + mm1) % nf1, (m2 + mm2) % nf2, (m3 + mm3) % nf3] \
                    += c * Em[mm1 + nspread, mm2 + nspread, mm3 + nspread]
    return ftau

#3d grid type 2 & type 1 with coil sensitivity kernel, sens_ker
#assume sens_ker matches the fntau data resolution, and size matches 2*nspread+1 for each dimension
@numba.jit(nopython=True, nogil=True)
def build_grid_3d21_sensker( x, y, z, fntau, tau, nspread, sens_ker = None ):
    if sens_ker is None: #then no coil sensitivity map will be used, but sill sens_ker need to be 4d
        sens_ker = np.ones((2 * nspread + 1, 2 * nspread + 1, 2 * nspread + 1, 1),\
                   dtype = fntau.dtype)/((2 * nspread + 1)**3)
    nc    = sens_ker.shape[3] #currently fix the forth dim as the coil dim
    nf1   = fntau.shape[0]    #in addtion, the forth dim of fntau need to be 1 or None
    nf2   = fntau.shape[1]
    nf3   = fntau.shape[2]
    hx    = 2 * np.pi / nf1
    hy    = 2 * np.pi / nf2
    hz    = 2 * np.pi / nf3
    Em    = np.zeros((2 * nspread + 1, 2 * nspread + 1, 2 * nspread + 1)) #will reuse this exponential
    # make the forth dim be the coil dim, which need to be length = 1 for fntau
    if len(fntau.shape) > 3 and fntau.shape[3] > 1: #if dim > 3, and dim 4 (coil dim) is not 1
        # input dim [nf1, nf2, nf3, a ,b c] output [nf1, nf2, nf3, 1, a, b, c]
        fntau     = np.expand_dims(fntau, axis = 3)
    #zero pad the sens_ker, enforce the sens_ker equals to convlution kernel for simplicity
    if sens_ker.shape[0:3] != (2 * nspread + 1, 2 * nspread + 1, 2 * nspread + 1 ):
        sens_ker  = ut.pad_or_cut3d( sens_ker, 2 * nspread + 1, 2 * nspread + 1, 2 * nspread + 1 )
    #initial ftau as zeros for output data
    ftau          = np.zeros(fntau.shape, dtype = fntau.dtype)
    outftaushape, outsenskshape = ut.dim_match(fntau.shape, sens_ker.shape)#match the dim, by adding 1 in extra dim
    fntau         = fntau.reshape(outftaushape)# [nf1, nf2, nf3, 1, a, b, c] or [nf1, nf2, nf3, 1] if 3d data
    ftau          = ftau.reshape(outftaushape) # [nf1, nf2, nf3, 1, a, b, c] or [nf1, nf2, nf3, 1] if 3d data
    sens_ker      = sens_ker.reshape(outsenskshape) #[nk, nk, nk, nc, 1, 1, 1] or [nk, nk, nk, nc] nk = 2*nspread + 1
    #do gridding for each ksp data point
    for i in range(x.shape[0]):
        c  = np.multiply(np.zeros(fntau[0,0,0].shape, dtype = fntau.dtype), \
             np.zeros(sens_ker[0,0,0].shape, dtype = fntau.dtype)) #coefficient, saved temporarily
        xi = x[i] % (2 * np.pi) #x, shift the source point xj so that it lies in [0,2*pi]
        yi = y[i] % (2 * np.pi) #y, shift the source point yj so that it lies in [0,2*pi]
        zi = z[i] % (2 * np.pi) #z, shift the source point zj so that it lies in [0,2*pi]
        m1 = 1 + int(xi // hx) #index for the closest grid point
        m2 = 1 + int(yi // hy) #index for the closest grid point
        m3 = 1 + int(zi // hz) #index for the closest grid point
        #grid once
        for mm1 in range(-nspread, nspread): #mm index for all the spreading points
            for mm2 in range(-nspread,nspread):
                for mm3 in range(-nspread,nspread):
                    #griding with g(x,y) = exp(-(x^2 + y^2 + z^2) / 4*tau)
                    Em[mm1 + nspread, mm2 + nspread, mm3 + nspread] =\
                     np.exp(-0.25 * (\
                    (xi - hx * (m1 + mm1)) ** 2 + \
                    (yi - hy * (m2 + mm2)) ** 2 + \
                    (zi - hz * (m3 + mm3)) ** 2 ) / tau)
                    #added sens_ker this is covlution with expoential Em and sens_kernel
                    c += np.multiply(\
                         fntau[(m1 + mm1) % nf1, (m2 + mm2) % nf2, (m3 + mm3) % nf3]\
                             * Em[mm1 + nspread, mm2 + nspread, mm3 + nspread], \
                         sens_ker[mm1 + nspread, mm2 + nspread, mm3 + nspread])
        #grid again
        c = c/(nf1*nf2*nf3)
        for mm1 in range(-nspread, nspread): #mm index for all the spreading points
            for mm2 in range(-nspread,nspread):
                for mm3 in range(-nspread,nspread):
                    #griding with g(x,y) = exp(-(x^2 + y^2 + z^2) / 4*tau)
                    #convlute with Em and conj(sens_ker), then sum along coil dimension
                    ftau[(m1 + mm1) % nf1, (m2 + mm2) % nf2, (m3 + mm3) % nf3] \
                    += np.sum(np.multiply(c * Em[mm1 + nspread, mm2 + nspread, mm3 + nspread],\
                                np.conj(sens_ker[mm1 + nspread, mm2 + nspread, mm3 + nspread])), axis = 3)
    return ftau
"""
main function of nufft3d
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
# 3d nufft type 1
#@numba.jit(nopython=True, nogil=True,parallel=True)
def nufft3d1_gaussker( x, y, z, c, ms, mt, mu, df=1.0, eps=1E-15, iflag=1, gridfast=1 ):
    """Fast Non-Uniform Fourier Transform with Numba"""
    nspread, nf1, nf2, nf3, tau = _compute_3d_grid_params(ms, mt, mu, eps)
    #try to override nspread
    nspread = min(3, nspread)
    #compute Ftaushape
    if len(c.shape) > 1:
        Ftaushape = (nf1, nf2, nf3) + c.shape[1:len(c.shape)]
    else:
        Ftaushape = (nf1, nf2, nf3)
    #print(Ftaushape)
    # Construct the convolved grid
    if gridfast is 0:
        Ftau = build_grid_3d1(x * df, y * df, z *df, c, tau, nspread,\
                      np.zeros(Ftaushape, dtype=c.dtype))
    else:#precompute some exponentials, not working
        Ftau = build_grid_3d1_fast(x * df, y * df, z *df, c, tau, nspread,\
                      np.zeros(Ftaushape, dtype=c.dtype), \
                      np.zeros((nspread+1, nspread+1, nspread+1), dtype=c.dtype))

    # Compute the FFT on the convolved grid
    if iflag < 0:
        Ftau = (1 / (nf1 * nf2 * nf3)) * np.fft.fftn(Ftau,s=None,axes=(0,1,2))
    else:
        Ftau = np.fft.ifftn(Ftau,s=None,axes=(0,1,2))
    #ut.plotim3(np.absolute(Ftau[:,:,:]))
    #truncate the Ftau to match the size of output, alias are removed
    Ftau = np.concatenate([Ftau[-(ms//2):,:,:], Ftau[:ms//2 + ms % 2,:,:]],0)
    Ftau = np.concatenate([Ftau[:,-(mt//2):,:], Ftau[:,:mt//2 + mt % 2,:]],1)
    Ftau = np.concatenate([Ftau[:,:,-(mu//2):], Ftau[:,:,:mu//2 + mu % 2]],2)
    # Deconvolve the grid using convolution theorem, Ftau * G(k1,k2,k3)^-1
    k1, k2, k3 = nufftfreqs3d(ms, mt, mu)
    # Note the np.sqrt(np.pi / tau)**3 due to the 3 dimentions of nufft
    outkshape, outFkshape = ut.dim_match(k1.shape ,Ftau.shape)
    return (1 / len(x)) * np.sqrt(np.pi / tau)**3 * \
    np.multiply(np.exp(tau * (k1 ** 2 + k2 ** 2 + k3 ** 2)).reshape(outkshape), Ftau.reshape(outFkshape))

#3d unfft type 2
def nufft3d2_gaussker( x, y, z, Fk, ms, mt, mu, df=1.0, eps=1E-15, iflag=1, gridfast=0 ):
    """Fast Non-Uniform Fourier Transform with Numba"""
    nspread, nf1, nf2, nf3, tau = _compute_3d_grid_params(ms, mt, mu, eps)

    # Deconvolve the grid using convolution theorem, Ftau * G(k1)^-1
    k1, k2, k3 = nufftfreqs3d(ms, mt, mu)
    # Note the np.sqrt(np.pi / tau)**3 due to the 3 dimentions of nufft
    outkshape, outFkshape = ut.dim_match(k1.shape ,Fk.shape)
    Fk = np.sqrt(np.pi / tau)**3 \
        * np.multiply(np.exp(tau * (k1 ** 2 + k2 ** 2 + k3 ** 2)).reshape(outkshape), Fk.reshape(outFkshape))
    #compute Ftau shape
    if len(Fk.shape) > 3:
        Ftaushape = (nf1, nf2, nf3) + Fk.shape[3:len(Fk.shape)]
    else:
        Ftaushape = (nf1, nf2, nf3)
    #reshape Fk and fftshift to match the size Fntau or Ftau
    Fntau = np.zeros(Ftaushape, dtype=Fk.dtype)
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
        Fntau = (nf1 * nf2 * nf3) * np.fft.ifftn(Fntau,s=None,axes=(0,1,2))
    else:
        Fntau = np.fft.fftn(Fntau,s=None,axes=(0,1,2))

    # Construct the convolved grid
    if gridfast is not 1:
        fx = build_grid_3d2(x*df, y*df, z*df, Fntau, tau, nspread)
    else:
        fx = build_grid_3d2_fast(x*df, y*df, z*df, Fntau, tau, nspread,\
         np.zeros((nspread+1, nspread+1, nspread+1), dtype=Fk.dtype))
    return fx

#3d unfft type 2
def nufft3d21_gaussker( x, y, z, Fk, ms, mt, mu, df=1.0, eps=1E-15, iflag=1, gridfast=1 ):
    """Fast Non-Uniform Fourier Transform with Numba"""
    nspread, nf1, nf2, nf3, tau = _compute_3d_grid_params(ms, mt, mu, eps)

    # Deconvolve the grid using convolution theorem, Ftau * G(k1)^-1
    k1, k2, k3 = nufftfreqs3d(ms, mt, mu)
    # Note the np.sqrt(np.pi / tau)**3 due to the 3 dimentions of nufft
    outkshape, outFkshape = ut.dim_match(k1.shape ,Fk.shape)
    Fk = np.sqrt(np.pi / tau)**3 \
        * np.multiply(np.exp(tau * (k1 ** 2 + k2 ** 2 + k3 ** 2)).reshape(outkshape), Fk.reshape(outFkshape))

    #compute Ftau shape
    if len(Fk.shape) > 3:
        Ftaushape = (nf1, nf2, nf3) + Fk.shape[3:len(Fk.shape)]
    else:
        Ftaushape = (nf1, nf2, nf3)
    #reshape Fk and fftshift to match the size Fntau or Ftau
    Ftau = np.zeros(Ftaushape, dtype=Fk.dtype)
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
        Ftau = (nf1 * nf2 * nf3) * np.fft.ifftn(Ftau,s=None,axes=(0,1,2))
    else:
        Ftau = np.fft.fftn(Ftau,s=None,axes=(0,1,2))

    # Construct the convolved grid
    if 1:# gridfast is not 1:
        Ftau = build_grid_3d21(x*df, y*df, z*df, Ftau, tau, nspread)
    #else:
    #    fx = build_grid_3d21_fast(x/df, y/df, z/df, fntau, tau, nspread,\
    #     np.zeros((nspread+1, nspread+1, nspread+1), dtype=Fk.dtype))

    # Compute the FFT on the convolved grid
    if iflag < 0:
        Ftau = (1 / (nf1 * nf2 * nf3)) * np.fft.fftn(Ftau,s=None,axes=(0,1,2))
    else:
        Ftau = np.fft.ifftn(Ftau,s=None,axes=(0,1,2))
    #ut.plotim3(np.absolute(Ftau[:,:,:]))
    #truncate the Ftau to match the size of output, alias are removed
    Ftau = np.concatenate([Ftau[-(ms//2):,:,:], Ftau[:ms//2 + ms % 2,:,:]],0)
    Ftau = np.concatenate([Ftau[:,-(mt//2):,:], Ftau[:,:mt//2 + mt % 2,:]],1)
    Ftau = np.concatenate([Ftau[:,:,-(mu//2):], Ftau[:,:,:mu//2 + mu % 2]],2)
    # Deconvolve the grid using convolution theorem, Ftau * G(k1,k2,k3)^-1
    # Note the np.sqrt(np.pi / tau)**3 due to the 3 dimentions of nufft
    return (1 / len(x)) * np.sqrt(np.pi / tau)**3 * \
    np.multiply(np.exp(tau * (k1 ** 2 + k2 ** 2 + k3 ** 2)).reshape(outkshape), Ftau.reshape(outFkshape))


#def test():
    #test nudft
    #nufft_test_func.time_nufft1d1(nudft1d1, 16, 1280)
    #nufft_test_func.time_nufft2d1(nudft2d1,64,64,5120)
    #nufft_test_func.time_nufft3d1(nudft3d1,32,32,16,2048)

    #test nufft type1
    #nufft_test_func.time_nufft1d1(nufft1d1_gaussker,64,512000,5)
    #nufft_test_func.time_nufft2d1(nufft2d1_gaussker,64,64,5120)
    #nufft_test_func.time_nufft3d1(nufft3d1_gaussker,32,32,16,2048)

    #test nufft type2
    #nufft_test_func.time_nufft1d2(nufft1d1_gaussker,nufft1d2_gaussker,32,102400,10)
    #nufft_test_func.time_nufft2d2(nufft2d1_gaussker,nufft2d2_gaussker,16,16,25000,1)
    #nufft_test_func.time_nufft3d2(nufft3d1_gaussker,nufft3d2_gaussker,16,16,8,20480,1)

    #compare
    #nufft_test_func.compare_nufft1d1(nudft1d1, nufft1d1_gaussker, 32, 12800)
    #nufft_test_func.compare_nufft2d1(nudft2d1, nufft2d1_gaussker, 16, 16,25000)
    #nufft_test_func.compare_nufft3d1(nudft3d1, nufft3d1_gaussker, 32, 32,16,20480)

    #compare type 2& typ1
    #nufft_test_func.compare_nufft1d21(nufft1d1_gaussker, nufft1d21_gaussker, 128, 100000,1)
    #nufft_test_func.compare_nufft2d21(nufft2d1_gaussker, nufft2d21_gaussker,16,16,25000,1)
    #nufft_test_func.compare_nufft3d21(nufft3d1_gaussker, nufft3d21_gaussker,16,16,8,20480,1)
#if __name__ == "__main__":
#    test()
