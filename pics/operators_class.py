import numpy as np
import dwt.dwt_func as dwt_func
import scipy.io as sio
#from fft.cufft import fftnc2c_cuda, ifftnc2c_cuda
import fft.fftw_func as fftw
from utilities.utilities_func import dim_match
class data_class:
    def __init__( self, data, dims_name ):
        self.dims_name = dims_name
        self.data = data

# match the dimensions of A and B, by adding 1 
# A_shape and B_shape are tuples from e.g. A.shape and B.shape  
"""
def dim_match( A_shape ,B_shape ):
    #intialize A_out_shape, B_out_shape
    A_out_shape = A_shape
    B_out_shape = B_shape
    #match them by adding 1
    if   len(A_shape) < len(B_shape):            
        for _ in range(len(A_shape),len(B_shape)):
            A_out_shape += (1,)
    elif len(A_shape) > len(B_shape):
        for _ in range(len(B_shape),len(A_shape)):
            B_out_shape += (1,)
    return  A_out_shape, B_out_shape
"""
"""
these classes apply  FFT for the input image,
 and some also apply mask in the forward function
the order is 
k-space -> image for forward; 
image -> k-space is backward
"""
#2d fft
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

#2d fft with mask
class FFT2d_kmask:
    "this is 2d FFT with k-space mask for CS MRI recon"
    def __init__( self, mask, axes=(0,1) ):
        self.mask = mask #save the k-space mask
        self.axes = axes        
    # let's call k-space <- image as forward
    def forward( self, im ):
        im   = np.fft.fftshift(im,self.axes)         
        ksp  = np.fft.fft2(im,s=None,axes=self.axes)
        ksp  = np.fft.ifftshift(ksp,self.axes)
        #try to match the dims of ksp and mask
        if len(ksp.shape) is not len(self.mask.shape):
            #try to match the dimensions of ksp and mask
            ksp_out_shape, mask_out_shape = dim_match(ksp.shape, self.mask.shape)
            mksp = np.multiply(ksp.reshape(ksp_out_shape),\
                         self.mask.reshape(mask_out_shape))#apply mask
        else:
            mksp = np.multiply(ksp,self.mask)#apply mask
        return mksp

    # let's call image <- k-space as backward
    def backward( self, ksp ):
        """
        #try to match the dims of ksp and mask
        if len(ksp.shape) is not len(self.mask.shape):
            #try to match the dimensions of ksp and mask
            ksp_out_shape, mask_out_shape = dim_match(ksp.shape, self.mask.shape)
            mksp = np.multiply(ksp.reshape(ksp_out_shape),\
                         self.mask.reshape(mask_out_shape))#apply mask
        else:
            mksp = np.multiply(ksp,self.mask)#apply mask
        """
        ksp = np.fft.fftshift(ksp,self.axes)
        im = np.fft.ifft2(ksp,s=None,axes=self.axes)
        #im = np.fft.ifft2(ksp,s=None,axes=self.axes)
        im = np.fft.ifftshift(im,self.axes)
        return im

#nd fft, default is 3d
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

#nd fft with mask
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
        #return np.multiply(ksp,self.mask)
        #try to match the dims of ksp and mask
        if len(ksp.shape) is not len(self.mask.shape):
            #try to match the dimensions of ksp and mask
            ksp_out_shape, mask_out_shape = dim_match(ksp.shape, self.mask.shape)
            mksp = np.multiply(ksp.reshape(ksp_out_shape),\
                         self.mask.reshape(mask_out_shape))#apply mask
        else:
            mksp = np.multiply(ksp,self.mask)#apply mask
        return mksp

    # let's call image <- k-space as backward
    def backward( self, ksp ):
        """
        #try to match the dims of ksp and mask
        if len(ksp.shape) is not len(self.mask.shape):
            #try to match the dimensions of ksp and mask
            ksp_out_shape, mask_out_shape = dim_match(ksp.shape, self.mask.shape)
            mksp = np.multiply(ksp.reshape(ksp_out_shape),\
                         self.mask.reshape(mask_out_shape))#apply mask
        else:
            mksp = np.multiply(ksp,self.mask)#apply mask
        """
        ksp = np.fft.fftshift(ksp,self.axes)
        #im = np.fft.ifftn(ksp,s=None,axes=self.axes)
        im  = np.fft.ifftn(ksp,s=None,axes=self.axes)
        im  = np.fft.ifftshift(im,self.axes)          
        return im

"""
those classes use fftw lib wihich support multi-threads
"""
#2d fft
class FFTW2d:
    "this is ndim FFTW for CS MRI recon"
    def __init__( self, axes = (0,1), threads = 1 ):
        self.axes = axes
        self.threads = threads

    # let's call k-space <- image as forward
    def forward( self, im):
        im  = np.fft.fftshift(im,self.axes)         
        ksp = fftw.fftw2d(im, axes=self.axes, threads = self.threads)
        ksp = np.fft.ifftshift(ksp,self.axes)
        return ksp

    # let's call image <- k-space as backward
    def backward( self, ksp ):
        ksp = np.fft.fftshift(ksp,self.axes)
        im  = fftw.ifftw2d(ksp,axes=self.axes, threads = self.threads)
        im  = np.fft.ifftshift(im,self.axes)          
        return im

#2d fft with mask
class FFTW2d_kmask:
    "this is 2dim FFTW with k-space mask for CS MRI recon"
    def __init__( self, mask, axes = (0,1), threads = 1 ):
        self.mask = mask #save the k-space mask
        self.axes = axes
        self.threads = threads

    # let's call k-space <- image as forward
    def forward( self, im):
        im  = np.fft.fftshift(im,self.axes)         
        ksp = fftw.fftw2d(im, axes=self.axes, threads = self.threads)
        ksp = np.fft.ifftshift(ksp,self.axes)
        #return np.multiply(ksp,self.mask)
        #try to match the dims of ksp and mask
        if len(ksp.shape) is not len(self.mask.shape):
            #try to match the dimensions of ksp and mask
            ksp_out_shape, mask_out_shape = dim_match(ksp.shape, self.mask.shape)
            mksp = np.multiply(ksp.reshape(ksp_out_shape),\
                         self.mask.reshape(mask_out_shape))#apply mask
        else:
            mksp = np.multiply(ksp,self.mask)#apply mask
        return mksp

    # let's call image <- k-space as backward
    def backward( self, ksp ):
        """
        #try to match the dims of ksp and mask
        if len(ksp.shape) is not len(self.mask.shape):
            #try to match the dimensions of ksp and mask
            ksp_out_shape, mask_out_shape = dim_match(ksp.shape, self.mask.shape)
            mksp = np.multiply(ksp.reshape(ksp_out_shape),\
                         self.mask.reshape(mask_out_shape))#apply mask
        else:
            mksp = np.multiply(ksp,self.mask)#apply mask
        """
        ksp = np.fft.fftshift(ksp,self.axes)
        im  = fftw.ifftw2d(ksp,axes=self.axes, threads = self.threads)
        im  = np.fft.ifftshift(im,self.axes)          
        return im


#nd fft
class FFTWnd:
    "this is ndim FFTW with k-space mask for CS MRI recon"
    def __init__( self, axes = (0,1,2), threads = 1 ):
        self.axes = axes
        self.threads = threads

    # let's call k-space <- image as forward
    def forward( self, im):
        im  = np.fft.fftshift(im,self.axes)         
        ksp = fftw.fftwnd(im, axes=self.axes, threads = self.threads)
        ksp = np.fft.ifftshift(ksp,self.axes)
        return ksp

    # let's call image <- k-space as backward
    def backward( self, ksp ):
        ksp = np.fft.fftshift(ksp,self.axes)
        im  = fftw.ifftwnd(ksp,axes=self.axes, threads = self.threads)
        im  = np.fft.ifftshift(im,self.axes)          
        return im

#nd fft with mask
class FFTWnd_kmask:
    "this is ndim FFTW with k-space mask for CS MRI recon"
    def __init__( self, mask, axes = (0,1,2), threads = 1 ):
        self.mask = mask #save the k-space mask
        self.axes = axes
        self.threads = threads

    # let's call k-space <- image as forward
    def forward( self, im):
        im  = np.fft.fftshift(im,self.axes)         
        ksp = fftw.fftwnd(im, axes=self.axes, threads = self.threads)
        ksp = np.fft.ifftshift(ksp,self.axes)
        #return np.multiply(ksp,self.mask)
        #try to match the dims of ksp and mask
        if len(ksp.shape) is not len(self.mask.shape):
            #try to match the dimensions of ksp and mask
            ksp_out_shape, mask_out_shape = dim_match(ksp.shape, self.mask.shape)
            mksp = np.multiply(ksp.reshape(ksp_out_shape),\
                         self.mask.reshape(mask_out_shape))#apply mask
        else:
            mksp = np.multiply(ksp,self.mask)#apply mask
        return mksp

    # let's call image <- k-space as backward
    def backward( self, ksp ):
        """
        #try to match the dims of ksp and mask
        if len(ksp.shape) is not len(self.mask.shape):
            #try to match the dimensions of ksp and mask
            ksp_out_shape, mask_out_shape = dim_match(ksp.shape, self.mask.shape)
            mksp = np.multiply(ksp.reshape(ksp_out_shape),\
                         self.mask.reshape(mask_out_shape))#apply mask
        else:
            mksp = np.multiply(ksp,self.mask)#apply mask
        """
        ksp = np.fft.fftshift(ksp,self.axes)
        im  = fftw.ifftwnd(ksp,axes=self.axes, threads = self.threads)
        im  = np.fft.ifftshift(im,self.axes)          
        return im


"""
discrete wavelet transform operators
"""
#2d dwt
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

## nd dwt
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

class espirit:
    "this is coil sensitivity operator"
    def __init__( self, sensitivity = None, coil_axis = None ):
        self.sens = sensitivity
        if coil_axis is None and self.sens is not None:
            # last dim of sensitivity map is coil axis
            self.coil_axis = len(sensitivity.shape)-1
        else:
            self.coil_axis = coil_axis

    #  apply coil combination
    def backward( self, im_coils ):  
        sens_out_shape, im_out_shape = dim_match(self.sens.shape,im_coils.shape)
        # coil combination is sum(conj(sens)*im)
        return np.sum(np.multiply(im_coils.reshape(im_out_shape),\
                     np.conj(self.sens).reshape(sens_out_shape))\
                    , axis=self.coil_axis, keepdims = True)

    # multiply image with coil sensitivity profile
    def forward( self, im_sos ):
        sens_out_shape, im_out_shape = dim_match(self.sens.shape,im_sos.shape)
        #appying sensitivity profile is sens*im
        return np.multiply(im_sos.reshape(im_out_shape),\
                        self.sens.reshape(sens_out_shape))
    
    #define save function
    def save( self, name ):
        sio.savemat(name, {'sens': self.sens, 'coil_axis': self.coil_axis})
        return self

    #restore sensitivity map
    def restore( self, name ):
        mat_contents   = sio.loadmat(name); 
        self.sens      = mat_contents['sens']
        self.coil_axis = np.int_(mat_contents['coil_axis'])
        return self


# do nothing operator
class None_opt:
    "this apply nothing"
    def forward( self, xin ):
        return xin

    def backward( self, xin ):
        return xin

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
