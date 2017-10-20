import numpy as np
from fft.cufft import fftnc2c_cuda, ifftnc2c_cuda, fft2c2c_cuda, ifft2c2c_cuda
import fft.nufft_func_cuda as nft_cuda
from utilities.utilities_func import dim_match
#import skcuda

class FFTnd_cuda_kmask:
    "this is ndim FFT_cuda with k-space mask for CS MRI recon"
    def __init__( self, mask, axes = (0,1,2)):
        self.mask = mask #save the k-space mask
        self.axes = axes
        #skcuda.misc.init()

    # let's call k-space <- image as forward
    def forward( self, im ):
        im  = np.fft.fftshift(im,self.axes)    
        ksp = fftnc2c_cuda(im)
        ksp = np.fft.ifftshift(ksp,self.axes)
        return np.multiply(ksp,self.mask)

    # let's call image <- k-space as backward
    def backward( self, ksp ):
        ksp = np.fft.fftshift(ksp,self.axes)
        #im = ifftnc2c_cuda(ksp)
        im = ifftnc2c_cuda(ksp)
        im = np.fft.ifftshift(im,self.axes)  
        return im

class FFT2d_cuda_kmask:
    "this is ndim FFT_cuda with k-space mask for CS MRI recon"
    def __init__( self, mask, axes = (0,1)):
        self.mask = mask #save the k-space mask
        self.axes = axes
        #skcuda.misc.init()

    # let's call k-space <- image as forward
    def forward( self, im ):
        im  = np.fft.fftshift(im,self.axes)    
        ksp = fft2c2c_cuda(im,self.axes)
        ksp = np.fft.ifftshift(ksp,self.axes)
        return np.multiply(ksp,self.mask)

    # let's call image <- k-space as backward
    def backward( self, ksp ):
        ksp = np.fft.fftshift(ksp,self.axes)
        #im = ifftnc2c_cuda(ksp)
        im = ifft2c2c_cuda(ksp,self.axes)
        im = np.fft.ifftshift(im,self.axes)  
        return im


"""
 NUFFT3d operators, apply to mutliple coil array data as a whole
"""
class NUFFT3d_cuda:
    def __init__( self, im_shape, dcf, ktrajx=None, ktrajy=None, ktrajz =None, axes = (0, 1, 2)):
        self.axes     = axes
        if ktrajx is not None:
            self.ktrajx   = ktrajx.flatten()
        else:
            self.ktrajx   = None
        if ktrajy is not None:
            self.ktrajy   = ktrajy.flatten()
        else:
            self.ktrajy   = None
        if ktrajz is not None:
            self.ktrajz   = ktrajz.flatten()
        else:
            self.ktrajz   = None
        self.im_shape = im_shape
        if dcf is not None:
            self.dcf      = dcf.flatten()
        else:
            self.dcf      = None

    def normalize_set_ktraj( self, ktraj ):
        ktraj           = np.pi * ktraj/ktraj.flatten().max()#2.0 * 
        self.ktrajx     = ktraj[0,:].flatten()
        self.ktrajy     = ktraj[1,:].flatten()
        self.ktrajz     = ktraj[2,:].flatten()#*(1.0*im_shape[0]/im_shape[2])    
        return self.ktrajx, self.ktrajy, self.ktrajz    

    def set_ktraj( self, ktrajx, ktrajy, ktrajz, dcf ):
        self.ktrajx   = ktrajx.flatten()
        self.ktrajy   = ktrajy.flatten()
        self.ktrajz   = ktrajz.flatten()
        self.dcf      = dcf.flatten()

    #def flatten_dims( self, kdata, axes = (0, 1, 2) ):
    #    return kdata.reshape((np.prod(kdata.shape[axes]),) + kdata.shape[axes[-1]:]).squeeze()


    def density_weighting( self, kdata, dcf ):
        kdatashape, dcfshape = dim_match(kdata.shape, dcf.shape)
        kdata                = np.multiply(kdata.reshape(kdatashape), dcf.reshape(dcfshape)).squeeze()
        return kdata

    def forward( self, im ):
        kdata    = nft_cuda.nufft3d2_gaussker_cuda(self.ktrajx,      self.ktrajy,      self.ktrajz,     im, \
                                                   self.im_shape[0], self.im_shape[1], self.im_shape[2], \
                                                   df=1.0, eps=1E-5, gridfast=1 )
        return kdata

    def backward( self, kdata ):
        # density conpensation
        kdata = self.density_weighting(kdata, self.dcf)
        # nufft type 1,
        im    = nft_cuda.nufft3d1_gaussker_cuda(self.ktrajx,      self.ktrajy,      self.ktrajz,       kdata, \
                                                self.im_shape[0], self.im_shape[1], self.im_shape[2],\
                                                df=1.0, eps=1E-5,  gridfast=1)
        return im

    def forward_backward( self, im ):
        im    = nft_cuda.nufft3d21_gaussker_cuda(self.ktrajx,      self.ktrajy,      self.ktrajz,       im, \
                                                 self.im_shape[0], self.im_shape[1], self.im_shape[2], \
                                                 self.dcf, \
                                                 df=1.0, eps=1E-5, gridfast=1 )
        return im