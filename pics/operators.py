import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize as spopt
import scipy.fftpack as spfft



"""
this class apply 2d FFT for the input 2d image, and apply mask in the forward function
k-space -> image is forward; image -> k-space is backward
usage: 
im = np.ones((128,128))
mask = np.ones(im.shape)
fft2dm = FFT2d_kmask(mask)
ksp = fft2dm.forward(im)
imhat = fft2dm.backward(ksp)
"""
class FFT2d_kmask:
    "this is 2d FFT with k-space mask for CS MRI recon"
    def __intial__( self, mask ):
        self.mask = mask #save the k-space mask
    # let's call k-space <- image as forward
    def forward( self, im ):
        im = np.fft.ifftshift(im,(0,1))
        ksp = np.fft.ifft2(im,s=None,axes=(0,1))
        return np.multiply(ksp,self.mask)

    # let's call image <- k-space as backward
    def backward( self, ksp ):
        ksp = np.multiply(ksp,self.mask)
        im = np.fft.fft2(ksp,s=None,axes=(0,1))
        im = np.fft.fftshift(ksp,(0,1))
        return (im)
        

class FFT2d:
    "this is 2d FFT with k-space mask for CS MRI recon"
    #def __intial__( self ):
        #self.mask = mask #save the k-space mask

    # let's call k-space <- image as forward
    def forward( self, im ):
        im = np.fft.ifftshift(im,(0,1))
        ksp = np.fft.ifft2(im,s=None,axes=(0,1))
        return ksp

    # let's call image <- k-space as backward
    def backward( self, ksp ):
        im = np.fft.fft2(ksp,s=None,axes=(0,1))
        im = np.fft.fftshift(im,(0,1))
        return (im)

"""
this class appy coil conbation for 2d image
"""
"""
class SENSE2d:
    "this is coil sensitivity operator"
    def __intial__( self, sensitivity ):
        self.s = sensitivity

    def forward( self, im_coils ):
    #  apply coil conbination

    def backward( self, im_sos ):
    # multiply image with coil sensitivity profile
"""
"""
this class appy coil conbation for 2d image
"""
"""
class ESPIRiT:
    "this is coil sensitivity operator"
    def __intial__( self, sensitivity ):
        self.s = sensitivity

    def forward( self, im_coils ):
    #  apply coil conbination
    #np.inner

    def backward( self, im_sos ):
    # multiply image with coil sensitivity profile
"""

"""
this class combine two operators together, this is usefull for parallel imaging
usage:
im = np.ones((128,128))
sensitivity = np.ones((128,128,8))
mask = np.ones(im.shape)
fft2dm = FFT2d_kmask(mask)
sense2d = SENSE3d(sensitivity)

ft_sense = joint2operators(fft2dm,sense2d)

"""
class joint2operators:
    "this apply two operators jointly"
    def __inital__( self, Aopt, Bopt ):
        self.Aopt = Aopt
        self.Bopt = Bopt

    def forward( self, xin ):
        xout = self.Bopt.forward(self.Aopt.forward(xin))
        return xout

    def backward( self, xin ):
        xout = self.Aopt.backward(self.Bopt.backward(xin))
        return xout

"""
this class combine three operators together

"""
class joint3operators:
    "this apply two operators jointly"
    def __inital__( self, Aopt, Bopt, Copt ):
        self.Aopt = Aopt
        self.Bopt = Bopt
        self.Copt = Copt

    def forward( self, xin ):
        xout = self.Copt.forward(self.Bopt.forward(self.Aopt.forward(xin)))
        return xout

    def backward( self, xin ):
        xout = self.Copt.backward(self.Aopt.backward(self.Bopt.backward(xin)))
        return xout
