import numpy as np
import pics.proximal_func as pf
import pics.CS_MRI_solvers_func as solvers
import pics.proximal_func as fp
import pics.operators_class as opts
import utilities.utilities_func as ut
from espirit.espirit_func import espirit_2d
import scipy.io as sio
import test.MRI_recon.IDEAL_class as idealc
from skimage.restoration import unwrap_phase
#from unwrap import unwrap

def unwrap_freq( im ):
    #ra        = 5 * np.random.rand(1)[0]
    max_im    = ut.scaling(np.absolute(im))
    scaled_im = (im)/max_im*np.pi
    #ut.plotim1(im)
    im  = unwrap_phase(scaled_im.astype(np.float))/np.pi*max_im
    ut.plotim1(np.real(im),bar=1)
    return im

def test():
    # simulated image
    mat_contents = sio.loadmat('data/kellman_data/PKdata1.mat', struct_as_record=False, squeeze_me=True)
    xdata        = mat_contents["data"] 
    im           = xdata.images
    TE           = xdata.TE
    field        = xdata.FieldStrength
    fat_freq_arr = 42.58 * field * np.array([-3.80, -3.40, -2.60, -1.94, -0.39, 0.60])
    fat_rel_amp  = np.array([0.087, 0.693, 0.128, 0.004, 0.039, 0.048])
    print(TE)
    ut.plotim3(np.real(im[:,:,:]))
    nx,ny,nte = im.shape

    scaling = ut.scaling(im)
    b = im/scaling

    #ut.plotim3(mask)
    ut.plotim3(np.absolute(b)) #undersampled imag
    #parameters
    xpar         = np.zeros((nx,ny,3), np.complex128)

    # IDEAL and FFT jointly
    IDEAL = idealc.IDEAL_opt2(TE, fat_freq_arr , fat_rel_amp )#
    IDEAL.set_x(xpar) #should update in each gauss newtown iteration
    residual    = IDEAL.residual(b)
    #do L2 cs mri recon
    Nite  = 20 #number of iterations
    rho   = 1.0   
    ostep = 1.0 
    for i in range(20):
        dxpar = pf.prox_l2_Afxnb_CGD2( IDEAL.forward, IDEAL.backward, residual, rho, Nite )
        if i%5 == 0:
            ut.plotim3(np.absolute(xpar + ostep*dxpar)[...,0:2],bar=1)
            ut.plotim3(np.real(xpar + ostep*dxpar)[...,2],bar=1)
            ut.plotim3(np.imag(xpar + ostep*dxpar)[...,2],bar=1)
        xpar = xpar + ostep * dxpar#.astype(np.float64)   
        #xpar[:,:,2] = np.real(xpar[:,:,2])
        if i > 1: 
            xpar[:,:,2] = np.real(unwrap_freq(np.real(xpar[:,:,2])))\
            +1j*(np.imag(xpar[:,:,2]))
        IDEAL.set_x(xpar) #should update in each gauss newtown iteration
        residual    = IDEAL.residual(b)
    ut.plotim3(np.absolute(xpar)[...,0:2],bar=1)