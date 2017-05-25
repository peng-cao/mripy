import numpy as np
from fft.cufft import fftnc2c_cuda, ifftnc2c_cuda, fft2c2c_cuda, ifft2c2c_cuda
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

