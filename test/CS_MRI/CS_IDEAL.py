import numpy as np
import pics.proximal_func as pf
import pics.CS_MRI_solvers_func as solvers
import pics.proximal_func as fp
import pics.operators_class as opts
import pics.opt_alg as alg
import pics.tvop_class as tvopc
import utilities.utilities_func as ut
from espirit.espirit_func import espirit_2d
import scipy.io as sio
import test.MRI_recon.IDEAL_class as idealc


def test():
    # simulated image
    mat_contents = sio.loadmat('data/kellman_data/PKdata3.mat', struct_as_record=False, squeeze_me=True)
    xdata        = mat_contents["data"] 
    im           = xdata.images
    field        = xdata.FieldStrength
    b0_gain      = 100.0
    TE           = b0_gain * xdata.TE
    fat_freq_arr = (1.0/b0_gain) *  42.58 * field * np.array([-3.80, -3.40, -2.60, -1.94, -0.39, 0.60])
    fat_rel_amp  = np.array([0.087, 0.693, 0.128, 0.004, 0.039, 0.048])
 
    ut.plotim3(np.real(im[:,:,:]))
    nx,ny,nte  = im.shape
    #undersampling
    mask       = ut.mask3d( nx, ny, nte, [15,15,0], 0.8)
    #FTm   = opts.FFT2d_kmask(mask)
    FTm        = opts.FFTW2d_kmask(mask)
    #FTm   = opts.FFT2d()
    b          = FTm.forward(im)
    scaling    = ut.optscaling(FTm,b)
    b          = b/scaling

    #ut.plotim3(mask)
    ut.plotim3(np.absolute(FTm.backward(b))) #undersampled imag
    #parameters
    xpar        = np.zeros((nx,ny,3), np.complex128)
    #xpar[:,:,0]  = 10*np.ones((nx,ny))
    #ut.plotim3(np.absolute(xpar),[3,-1])
    # IDEAL and FFT jointly
    IDEAL       = idealc.IDEAL_opt2(TE, fat_freq_arr , fat_rel_amp )#fat_freq_arr , fat_rel_amp
    Aideal_ftm  = opts.joint2operators(IDEAL, FTm)#(FTm,IDEAL)#
    IDEAL.set_x(xpar) #should update in each gauss newton iteration
    residual    = IDEAL.residual(b, FTm)
    #ut.plotim3(np.absolute(FTm.backward(residual)))
    # wavelet and x+d_x
    addx        = idealc.x_add_dx()
    addx.set_x(xpar)
    #addx.set_w([1, 1, 0.0001])
    dwt         = opts.DWT2d(wavelet = 'haar', level=4)
    Adwt_addx   = opts.joint2operators(dwt, addx) 

    #do soft thresholding
    #Nite = 200 #number of iterations
    #step = 0.01 #step size
    #th   = 0.02 # theshold level
    #do tv cs mri recon
    Nite = 10 #number of iterations
    step = 1 #step size
    l1_r = 0.02
    tv_r = 0.001 # regularization term for tv term
    rho  = 1.0  
    ostep = 0.3 
      
    for i in range(200):
        #wavelet L1 IST
    #    dxpar = solvers.IST_3( Aideal_ftm.forward, Aideal_ftm.backward,\
    #                Adwt_addx.backward, Adwt_addx.forward, residual, Nite, step, th )
        #wavelet L1 ADMM
    #    dxpar = solvers.ADMM_l2Afxnb_l1Tfx( Aideal_ftm.forward, Aideal_ftm.backward, \
    #               Adwt_addx.backward, Adwt_addx.forward, residual, Nite, step, l1_r, rho, 200 )
        # TV ADMM
    #    dxpar = solvers.ADMM_l2Afxnb_tvx( Aideal_ftm.forward, Aideal_ftm.backward, residual\
    #    	, Nite, step, tv_r, rho, 15 ) 
        dxpar = solvers.ADMM_l2Afxnb_tvTfx( Aideal_ftm.forward, Aideal_ftm.backward, \
                   addx.backward, addx.forward, residual, Nite, step, l1_r, rho, 200 )

        # L2 CGD
    #    dxpar = pf.prox_l2_Afxnb_CGD2( Aideal_ftm.forward, Aideal_ftm.backward, residual, rho, Nite )
    #    dxpar = pf.prox_l2_Afxnb_CGD2( Aideal_ftm.forward, Aideal_ftm.backward, residual, Nite )

        if i%1 == 0:
            ut.plotim3(np.absolute(xpar + ostep*dxpar)[...,0:2],bar=1)
            ut.plotim3(b0_gain * np.real(xpar + ostep*dxpar)[...,2],bar=1)
            ut.plotim3(np.imag(xpar + ostep*dxpar)[...,2],bar=1)

        xpar = xpar + ostep*dxpar#.astype(np.float64)   

        IDEAL.set_x(xpar) #should update in each gauss newton iteration
        residual = IDEAL.residual(b, FTm)
        addx.set_x(xpar) #should update in each gauss newton iteration    
    ut.plotim3(np.absolute(xpar)[...,0:2],bar=1)