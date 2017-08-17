import numpy as np
import scipy.io as sio
import pics.operators_class as opts
import utilities.utilities_func as ut
import utilities.utilities_class as utc
from   signal_processing.filter_func import hamming2d, hamming3d
from   nufft_func import nufftfreqs2d, nufftfreqs3d
#####################################################################################################
#ft direct calculation for 1 spatial point, 2d,3d, type1
#####################################################################################################

#direct 2d FT, transfer k-space coefficients (k_c, array) to i point in image space at (ix,iy)
# ix, iy the position in image space
#kx, ky the grid in k-space
def dft2d_im1point( ix, iy, k_c, df=1.0, iflag=1 ):
    k_shape   = k_c.shape
    sign      = -1 if iflag < 0 else 1
    #grid in k-space
    kx, ky    = nufftfreqs2d(k_shape[0], k_shape[1], df)
    #kx, ky    = np.mgrid[0:k_shape[0], 0:k_shape[1]]
    kx = 1.0*kx/(k_shape[0])
    ky = 1.0*ky/(k_shape[1])
    k_i_prod  = np.multiply(k_c, np.exp(sign * 1j * 2.0 * np.pi * (\
        np.multiply(kx, ix) + np.multiply(ky, iy))))
    return (1.0/np.prod(k_shape[0:2])) * np.sum(k_i_prod)

#2d dft return image im, this function is design for testing purpose
def dft2d_warp( i_ms, i_mt, k_c, df=1.0, iflag=1 ):
    array_x = np.mgrid[-(i_ms//2):i_ms-(i_ms//2)]/(i_ms*1.0/k_c.shape[0])
    array_y = np.mgrid[-(i_mt//2):i_mt-(i_mt//2)]/(i_mt*1.0/k_c.shape[1])
    im      = np.zeros((i_ms,i_mt),dtype=k_c.dtype)
    for idx_x in range(i_ms):
        for idx_y in range(i_mt):
            im[idx_x,idx_y] = dft2d_im1point(array_x[idx_x], array_y[idx_y], k_c)
    return im


#direct 3d FT, transfer k-space coefficients (k_c, array) to i point in image space at (ix,iy)
# ix, iy, iz the position in image space
#kx, ky, kz the grid in k-space
def dft3d_im1point( ix, iy, iz, k_c, df=1.0, iflag=1 ):
    k_shape    = k_c.shape
    sign       = -1 if iflag < 0 else 1
    kx, ky, kz = nufftfreqs3d(k_shape[0], k_shape[1], k_shape[2], df)
    kx = 1.0*kx/(k_shape[0])
    ky = 1.0*ky/(k_shape[1])
    kz = 1.0*kz/(k_shape[2])
    k_i_prod   = np.multiply(k_c, np.exp(sign * 1j * 2.0 * np.pi * (\
        np.multiply(kx, ix) + np.multiply(ky, iy) + np.multiply(kz, iz))))
    
    return (1.0/np.prod(k_shape[0:3])) * sum(k_i_prod.flatten())#

#3d dft return image im, this function is design for testing purpose
def dft3d_warp( i_ms, i_mt, i_mu, k_c, df=1.0, iflag=1 ):
    array_x = np.mgrid[-(i_ms//2):i_ms-(i_ms//2)]/(i_ms*1.0/k_c.shape[0])
    array_y = np.mgrid[-(i_mt//2):i_mt-(i_mt//2)]/(i_mt*1.0/k_c.shape[1])
    array_z = np.mgrid[-(i_mu//2):i_mu-(i_mu//2)]/(i_mu*1.0/k_c.shape[2])
    im      = np.zeros((i_ms,i_mt,i_mu),dtype=k_c.dtype)
    for idx_x in range(i_ms):
        for idx_y in range(i_mt):
            for idx_z in range(i_mu):
                im[idx_x,idx_y,idx_z]\
                = dft3d_im1point(array_x[idx_x], array_y[idx_y], array_z[idx_z], k_c)
    return im

def test1():
    N         = 20
    k1, k2    = nufftfreqs2d(N, N)
    k_c       = np.cos(np.multiply(5,k1))+1j*np.sin(np.multiply(5,k2))#
    #k_c       = np.ones((N,N))#    
    hwin      = hamming2d(N,N)
    #ut.plotgray(np.absolute(hwin)) 
    # apply hamming window   
    k_c       = np.multiply(k_c, hwin)
    im        = dft2d_warp(N, N, k_c)
    ut.plotim1(np.absolute(im), bar = True)
    #use std fft lib
    ft        = opts.FFT2d()
    npim      = ft.backward(k_c)
    ut.plotim1(np.absolute(npim), bar = True)
    ut.plotim1(np.absolute(im-npim), bar = True)
    #interpolatation
    im_int    = dft2d_warp(5*N, 5*N, k_c)
    ut.plotim1(np.absolute(im_int), bar = True)    

def test2():
    N         = 20
    k1, k2, k3 = nufftfreqs3d(N, N, N)
    k_c        = np.cos(np.multiply(5,k1))+1j*np.sin(np.multiply(5,k2))#
    #k_c       = np.ones((N,N,N))#    
    hwin       = hamming3d(N,N,N)
    ut.plotim3(np.absolute(hwin),[4,-1]) 
    # apply hamming window   
    k_c       = np.multiply(k_c, hwin)
    im        = dft3d_warp(N, N, N, k_c)
    ut.plotim3(np.absolute(im),[4,-1], bar = True)
    #use std fft lib
    ft        = opts.FFTnd()
    npim      = ft.backward(k_c)
    ut.plotim3(np.absolute(npim),[4,-1], bar = True)
    ut.plotim3(np.absolute(im-npim),[4,-1], bar = True)
    #interpolatation
    im_int    = dft3d_warp(2*N, 2*N, N, k_c)
    ut.plotim3(np.absolute(im_int),[4,-1], bar = True) 
