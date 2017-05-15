import numpy as np
import pics.proximal_func as pf
import pics.CS_MRI_solvers_func as solvers
import pics.proximal_func as fp
import pics.operators_class as opts
import utilities.utilities_func as ut
from espirit.espirit_func import espirit_2d
import scipy.io as sio
import test.MRI_recon.IDEAL_class as idealc
def test():
    # simulated image
    mat_contents = sio.loadmat('data/kellman_data/PKdata3.mat', struct_as_record=False, squeeze_me=True)
    xdata        = mat_contents["data"] 
    im           = xdata.images
    TE           = xdata.TE
    field        = xdata.FieldStrength
    fat_freq_arr = 42.58 * field * np.array([-3.80, -3.40, -2.60, -1.94, -0.39, 0.60])
    fat_rel_amp  = np.array([0.087, 0.693, 0.128, 0.004, 0.039, 0.048])
 
    ut.plotim3(np.real(im[:,:,:]))
    nx,ny,nte = im.shape
    #undersampling
    #mask  = ut.mask3d( nx, ny, nte, [15,15,0], 0.8)
    #FTm   = opts.FFT2d_kmask(mask)
    FTm     = opts.FFT2d()
    b       = FTm.forward(im)
    scaling = ut.optscaling(FTm,b)
    b       = b/scaling

    #ut.plotim3(mask)
    ut.plotim3(np.absolute(FTm.backward(b))) #undersampled imag
    #parameters
    xpar         = np.zeros((nx,ny,3), np.complex128)

    # IDEAL and FFT jointly
    IDEAL       = idealc.IDEAL_opt2(TE, 217.0 , 1.0 )#fat_freq_arr , fat_rel_amp
    Aideal_ftm  = opts.joint2operators(IDEAL, FTm)#(FTm,IDEAL)#
    IDEAL.set_x(xpar) #should update in each gauss newtown iteration
    residual    = IDEAL.residual(b, FTm)
    #ut.plotim3(np.absolute(FTm.backward(residual)))
    # wavelet and x+d_x
    #addx        = idealc.x_add_dx()
    #addx.set_x(xpar) #should update in each gauss newtown iteration
    #dwt         = opts.DWT2d(wavelet = 'haar', level=4)
    #Adwt_addx   = opts.joint2operators(dwt, addx)

    #do L2 cs mri recon
    Nite  = 20 #number of iterations
    ostep = 1.0 
    for i in range(20):
        dxpar = pf.prox_l2_Afxnb_CGD2( Aideal_ftm.forward, Aideal_ftm.backward, residual, Nite )
        if i%5 == 0:
            ut.plotim3(np.absolute(xpar + ostep*dxpar)[...,0:2],bar=1)
            ut.plotim3(np.real    (xpar + ostep*dxpar)[...,2],bar=1)
            ut.plotim3(np.imag    (xpar + ostep*dxpar)[...,2],bar=1)
        xpar = xpar + ostep * dxpar#.astype(np.float64)   
        #xpar[:,:,2] = np.real(xpar[:,:,2])

        IDEAL.set_x(xpar) #should update in each gauss newtown iteration
        residual    = IDEAL.residual(b, FTm)
        #addx.set_x(xpar) #should update in each gauss newtown iteration

    ut.plotim3(np.absolute(xpar)[...,0:2],bar=1)