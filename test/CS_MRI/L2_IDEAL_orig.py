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
#pathdat = '/working/larson/UTE_GRE_shuffling_recon/IDEAL_ZTE/20170929/save_data1slice.mat'
pathdat = '/working/larson/UTE_GRE_shuffling_recon/IDEAL_ZTE/20171027/save_data.mat'

#datpath = './data/kellman_data/PKdata3.mat'

def unwrap_freq( im ):
    max_im    = 0.8*ut.scaling(np.absolute(im))
    scaled_im = (im)/max_im*np.pi
    ut.plotim1(im, bar = 1, pause_close = 5)
    im  = unwrap_phase(scaled_im)/np.pi*max_im
    ut.plotim1(im,bar=1, pause_close = 5)
    return im

def test():
    # simulated image
    mat_contents = sio.loadmat(pathdat, struct_as_record=False, squeeze_me=True)
    xdata        = mat_contents["data"] 
    im           = xdata.images
    TE           = xdata.TE
    field        = xdata.FieldStrength
    fat_freq_arr = 42.58 * field * np.array([-3.80, -3.40, -2.60, -1.94, -0.39, 0.60])
    fat_rel_amp  = np.array([0.087, 0.693, 0.128, 0.004, 0.039, 0.048])
    print(TE)
    #ut.plotim3(np.angle(im[:,:,:]))
    nx,ny,nz,nte = im.shape

    scaling = ut.scaling(im)
    b = im/scaling

    #ut.plotim3(mask)
    #ut.plotim3(np.absolute(b)) #undersampled imag
    #parameters
    xpar         = np.zeros((nx,ny,nz,3), np.complex128)

    # IDEAL and FFT jointly
    IDEAL = idealc.IDEAL_opt2(TE, fat_freq_arr , fat_rel_amp )#fat_freq_arr , fat_rel_amp
    IDEAL.set_x(xpar) #should update in each gauss newton iteration
    residual    = IDEAL.residual(b)
    #do L2 cs mri recon
    Nite  = 10 #number of iterations   
    ostep = 1.0 
    for i in range(40):
        dxpar = pf.prox_l2_Afxnb_CGD2( IDEAL.forward, IDEAL.backward, residual, Nite )
        #if i%1 == 0:
        #    ut.plotim3(np.absolute(xpar + ostep*dxpar)[...,0:2],bar=1, pause_close = 5)
        #    ut.plotim3(np.real(xpar + ostep*dxpar)[...,2],bar=1, pause_close = 5)
        #    ut.plotim3(np.imag(xpar + ostep*dxpar)[...,2],bar=1, pause_close = 5)
        xpar = xpar + ostep * dxpar#.astype(np.float64) 
        #xpar[...,2] = np.real(xpar[...,2])  
        #xpar[:,:,2] = np.real(xpar[:,:,2])
        #if i > 0: 
        #    xpar[:,:,2] = unwrap_freq(np.real(xpar[:,:,2]))\
        #    +1j*(np.imag(xpar[:,:,2]))
        IDEAL.set_x(xpar) #should update in each gauss newton iteration
        residual    = IDEAL.residual(b)
    #ut.plotim3(np.absolute(xpar)[...,0:2],bar=1, pause_close = 5)
    sio.savemat(pathdat + 'IDEAL_org_result.mat', {'xpar':xpar, 'residual':residual})