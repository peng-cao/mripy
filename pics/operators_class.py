import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize as spopt
import scipy.fftpack as spfft
import dwt.dwt_func as dwt_func
#from fft.cufft import fftnc2c_cuda, ifftnc2c_cuda

class data_class:
    def __init__( self, data, dims_name ):
        self.dims_name = dims_name
        self.data = data

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
    def __init__( self, mask, axes=(0,1) ):
        self.mask = mask #save the k-space mask
        self.axes = axes        
    # let's call k-space <- image as forward
    def forward( self, im ):
        im = np.fft.fftshift(im,self.axes)         
        ksp = np.fft.fft2(im,s=None,axes=self.axes)
        ksp = np.fft.ifftshift(ksp,self.axes)
        return np.multiply(ksp,self.mask)#apply mask

    # let's call image <- k-space as backward
    def backward( self, ksp ):
        ksp = np.fft.fftshift(ksp,self.axes)
        im = np.fft.ifft2(ksp,s=None,axes=self.axes)
        #im = np.fft.ifft2(ksp,s=None,axes=self.axes)
        im = np.fft.ifftshift(im,self.axes)
        return im


class FFT2d:
    "this is 2d FFT without k-space mask for CS MRI recon"
    def __init__( self, axes = (0,1)):
        #self.mask = mask #save the k-space mask
        self.axes = axes    
    # let's call k-space <- image as forward
    def forward( self, im ):
        im = np.fft.fftshift(im,self.axes)        
        ksp = np.fft.fft2(im,s=None,axes=self.axes)
        ksp = np.fft.ifftshift(ksp,self.axes)
        return ksp

    # let's call image <- k-space as backward
    def backward( self, ksp ):
        ksp = np.fft.fftshift(ksp,self.axes)
        #im = np.fft.ifft2(ksp,s=None,axes=(0,1))#noted that numpy.fft by default applies to last two dims
        im = np.fft.ifft2(ksp,s=None,axes=self.axes)
        im = np.fft.ifftshift(im,self.axes)
        return im

"""
define n-dim fft here, default is 3d
"""
class FFTnd:
    "this is ndim FFT without k-space mask for CS MRI recon"
    def __init__( self, axes = (0,1,2)):
        #self.mask = mask #save the k-space mask
        self.axes = axes

    # let's call k-space <- image as forward
    def forward( self, im ):
        im  = np.fft.fftshift(im,self.axes)                
        ksp = np.fft.fftn(im,s=None,axes=self.axes)
        ksp = np.fft.ifftshift(ksp,self.axes)        
        return ksp

    # let's call image <- k-space as backward
    def backward( self, ksp ):
        ksp = np.fft.fftshift(ksp,self.axes)
        #im = np.fft.ifftn(ksp,s=None,axes=self.axes) 
        im = np.fft.ifftn(ksp,s=None,axes=self.axes)
        im = np.fft.ifftshift(im,self.axes)               
        return im

class FFTnd_kmask:
    "this is ndim FFT with k-space mask for CS MRI recon"
    def __init__( self, mask, axes = (0,1,2)):
        self.mask = mask #save the k-space mask
        self.axes = axes

    # let's call k-space <- image as forward
    def forward( self, im ):
        im  = np.fft.fftshift(im,self.axes)         
        ksp = np.fft.fftn(im,s=None,axes=self.axes)
        ksp = np.fft.ifftshift(ksp,self.axes)
        return np.multiply(ksp,self.mask)

    # let's call image <- k-space as backward
    def backward( self, ksp ):
        ksp = np.fft.fftshift(ksp,self.axes)
        #im = np.fft.ifftn(ksp,s=None,axes=self.axes)
        im  = np.fft.ifftn(ksp,s=None,axes=self.axes)
        im  = np.fft.ifftshift(im,self.axes)          
        return im

class DWT2d:
    "this is 2d wavelet transform for CS MRI recon"
    def __init__( self, wavelet = 'db2', level = 2, axes = (0, 1) ):
        self.wavelet      = wavelet
        self.level        = level
        self.coeff_slices = []
        self.axes         = axes 
        self.shape        = None       
    # wavelet domain --> image, forward
    def forward( self, arr_coeff ):
        im  = dwt_func.idwt2d(arr_coeff, self.coeff_slices, self.wavelet, self.axes)
        if self.shape is not None:       
            for i in range(len(self.axes)):
                if self.shape[i] < im.shape[i]:#if shape mismatch, delete the last line 
                    im = np.delete(im,(im.shape[i]-1),axis=i)        
        return im
    # image --> wavelet domain, backward
    def backward( self, im ):
        if self.shape is None:
            self.shape = im.shape
        arr_coeff, self.coeff_slices  = dwt_func.dwt2d(im, self.wavelet, self.level, self.axes)
        #ut.plotim1(arr_coeff)
        return arr_coeff

class DWTnd:
    "this is nd wavelet transform for CS MRI recon"
    def __init__( self, wavelet = 'db2', level = 2, axes = (0, 1, 2) ):
        self.wavelet      = wavelet
        self.level        = level
        self.coeff_slices = []
        self.axes         = axes
        self.shape        = None
    # wavelet domain --> image, forward
    def forward( self, arr_coeff ):
        im         = dwt_func.idwtnd(arr_coeff, self.coeff_slices, self.wavelet, self.axes)
        if self.shape is not None:
            for i in range(len(self.axes)):
                if self.shape[i] < im.shape[i]:#if shape mismatch, delete the last line 
                    im = np.delete(im,(im.shape[i]-1),axis=i)
        return im
    # image --> wavelet domain, backward
    def backward( self, im ):
        if self.shape is None:
            self.shape = im.shape     
        arr_coeff, self.coeff_slices  = dwt_func.dwtnd(im, self.wavelet, self.level, self.axes)
        #ut.plotim1(arr_coeff)
        return arr_coeff

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
    def __init__( self, Aopt, Bopt ):
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
