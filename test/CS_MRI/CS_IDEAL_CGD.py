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

from skimage.restoration import unwrap_phase
#from unwrap import unwrap

def unwrap_freq( im ):
    max_im    = ut.scaling(np.absolute(im))
    scaled_im = (im)/max_im*np.pi
    #ut.plotim1(im)
    im  = unwrap_phase(scaled_im.astype(np.float))/np.pi*max_im
    ut.plotim1(np.real(im),bar=1)
    return im

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
    IDEAL       = idealc.IDEAL_opt2(TE, 217.0 , 1.0 )#fat_freq_arr , fat_rel_amp
    Aideal_ftm  = opts.joint2operators(IDEAL, FTm)#(FTm,IDEAL)#
    IDEAL.set_x(xpar) #should update in each gauss newtown iteration
    residual    = IDEAL.residual(b, FTm)
    #ut.plotim3(np.absolute(FTm.backward(residual)))
    # wavelet and x+d_x
    addx_water  = idealc.x_add_dx()
    addx_fat    = idealc.x_add_dx()
    addx_df     = idealc.x_add_dx()
    #addx        = idealc.x_add_dx()
    #addx.set_w([1, 1, 0.0001])
    addx_water.set_x(xpar[...,0]) #should update in each gauss newtown iteration
    addx_fat.set_x  (xpar[...,1])
    addx_df.set_x   (xpar[...,2])
    dwt         = opts.DWT2d(wavelet = 'haar', level=4)
    tvop        = tvopc.TV2d()
    Adwt_addx_w = opts.joint2operators(dwt, addx_water)   
    Adwt_addx_f = opts.joint2operators(dwt, addx_fat)
    Adwt_addx_d = opts.joint2operators(tvop, addx_df)   
    #Adwt_addx   = opts.joint2operators(dwt, addx)

    #CGD
    Nite  = 20
    l1_r1 = 0.1
    l1_r2 = 0.001
    def f(xi):
        #return np.linalg.norm(Aideal_ftm.forward(xi)-residual)
        return alg.obj_fidelity(Aideal_ftm, xi, residual) \
        + l1_r1 * alg.obj_sparsity(Adwt_addx_w, xi[...,0])\
        + l1_r1 * alg.obj_sparsity(Adwt_addx_f, xi[...,1])\
        + l1_r2 * alg.obj_sparsity(Adwt_addx_d, xi[...,2])

    def df(xi):
        #return 2*Aideal_ftm.backward(Aideal_ftm.forward(xi)-residual)
        gradall = alg.grad_fidelity(Aideal_ftm, xi, residual)
        gradall[...,0] += l1_r1 * alg.grad_sparsity(Adwt_addx_w, xi[...,0])
        gradall[...,1] += l1_r1 * alg.grad_sparsity(Adwt_addx_f, xi[...,1])
        gradall[...,2] += l1_r2 * alg.grad_sparsity(Adwt_addx_d, xi[...,2])  
        return gradall      

    #do soft thresholding
    #Nite = 20 #number of iterations
    #step = 0.1 #step size
    #th   = 1 # theshold level
    #do tv cs mri recon
    #Nite = 40 #number of iterations
    #step = 1 #step size
    #tv_r = 0.01 # regularization term for tv term
    #rho  = 1.0  
    ostep = 1.0#0.3       
    for i in range(20):
        #wavelet L1 IST
    #    dxpar = solvers.IST_3( Aideal_ftm.forward, Aideal_ftm.backward,\
    #                Adwt_addx.backward, Adwt_addx.forward, residual, Nite, step, th )
        #wavelet L1 ADMM
    #    dxpar = solvers.ADMM_l2Afxnb_l1Tfx( Aideal_ftm.forward, Aideal_ftm.backward, \
    #               Adwt_addx.backward, Adwt_addx.forward, residual, Nite, step, tv_r, rho,15 )
        # TV ADMM
    #    dxpar = solvers.ADMM_l2Afxnb_tvx( Aideal_ftm.forward, Aideal_ftm.backward, residual\
    #    	, Nite, step, tv_r, rho ) 
        # L2 CGD
    #    dxpar = pf.prox_l2_Afxnb_CGD2( Aideal_ftm.forward, Aideal_ftm.backward, residual, rho, Nite )
    #    dxpar = pf.prox_l2_Afxnb_CGD2( Aideal_ftm.forward, Aideal_ftm.backward, residual, Nite )
        # L1 CGD
        dxpar   = alg.conjugate_gradient(f, df, Aideal_ftm.backward(residual), Nite )
        #ostep,j = alg.BacktrackingLineSearch(f, df, xpar, dxpar)
        if i%1 == 0:
            ut.plotim3(np.absolute(xpar + ostep*dxpar)[...,0:2],bar=1)
            ut.plotim3(np.real(xpar + ostep*dxpar)[...,2],bar=1)
            ut.plotim3(np.imag(xpar + ostep*dxpar)[...,2],bar=1)
        xpar = xpar + ostep*dxpar#.astype(np.float64)   

        #if i > 1: #unwrapping on frequence
        #    xpar[:,:,2] = np.real(unwrap_freq(np.real(xpar[:,:,2])))\
        #    +1j*(np.imag(xpar[:,:,2]))

        IDEAL.set_x(xpar) #should update in each gauss newtown iteration
        residual = IDEAL.residual(b, FTm)
    #    addx.set_x(xpar) #should update in each gauss newtown iteration
        addx_water.set_x(xpar[...,0]) #should update in each gauss newtown iteration
        addx_fat.set_x  (xpar[...,1])
        addx_df.set_x   (xpar[...,2])
    ut.plotim3(np.absolute(xpar)[...,0:2],bar=1)
    ut.plotim3(np.real(xpar + ostep*dxpar)[...,2],bar=1)
    ut.plotim3(np.imag(xpar + ostep*dxpar)[...,2],bar=1)