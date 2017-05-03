import pywt
import numpy as np
import utilities.utilities_func as ut

def dwt2d( im, wavelet = 'db2', level = 2, axes = (0, 1) ):
    coeffs             = pywt.wavedec2(im, wavelet, level=level, axes=axes)
    arr, coeff_slices  = pywt.coeffs_to_array(coeffs)
    return arr, coeff_slices

def idwt2d( arr, coeff_slices, wavelet = 'db2', axes = (0,1) ):
    coeffs_from_arr    = pywt.array_to_coeffs(arr, coeff_slices)
    im                 = pywt.waverecn(coeffs_from_arr, wavelet, axes=axes)
    return im

def dwtnd( im, wavelet = 'db2', level = 2, axes = (0, 1, 2) ):
    coeffs             = pywt.wavedecn(im, wavelet, level=level, axes = axes)
    arr, coeff_slices  = pywt.coeffs_to_array(coeffs)
    return arr, coeff_slices

def idwtnd( arr, coeff_slices, wavelet = 'db2', axes = (0, 1, 2) ):
    coeffs_from_arr    = pywt.array_to_coeffs(arr, coeff_slices)
    im                 = pywt.waverecn(coeffs_from_arr, wavelet, axes = axes)
    return im

def test():
    cam                = pywt.data.camera()
    ut.plotim1(cam)
    arr, coeff_slices  = dwt2d(cam)
    ut.plotim1(arr)
    cam_recon          = idwt2d(arr, coeff_slices)
    ut.plotim1(cam_recon)
    #np.testing.assert_array_almost_equal(cam,cam_recon)

