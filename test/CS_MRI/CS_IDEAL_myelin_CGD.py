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
import h5py
def unwrap_freq( im ):
    max_im    = ut.scaling(np.absolute(im))
    scaled_im = (im)/max_im*np.pi
    #ut.plotim1(im)
    im  = unwrap_phase(scaled_im.astype(np.float))/np.pi*max_im
    ut.plotim1(np.real(im),bar=1)
    return im

def test():
    # simulated image
    #mat_contents = sio.loadmat('/data/larson/brain_uT2/2016-09-13_3T-volunteer/ute_32echo_random-csreconallec_l2_r0p01.mat', struct_as_record=False, squeeze_me=True)

    #datpath = '/data/larson/brain_uT2/2016-12-22_7T-volunteer/' #
    datpath = '/data/larson/brain_uT2/2016-09-13_3T-volunteer/'
    f = h5py.File(datpath+'ute_32echo_random-csreconallec_l2_r0p01.mat')

    #im3d         = f['imallplus'][0:10].transpose([1,2,3,0])
    #im           = im3d[:,40,:,:].squeeze().view(np.complex128)
    Ndiv         = 8
    im3d         = f['imallplus'][0:Ndiv].transpose([1,3,2,0])
    im           = im3d[35,:,:,:].squeeze().view(np.complex128)

    b0_gain      = 1000.0
    TE           = b0_gain * 1e-6 * f['TE'][0][0:Ndiv]
    field        = 3.0
    fat_freq_arr = (1.0/b0_gain) * 42.58 * field * np.array([-3.80, -3.40, -2.60, -1.94, -0.39, 0.60])
    fat_rel_amp  = np.array([0.087, 0.693, 0.128, 0.004, 0.039, 0.048])
    print(1000/b0_gain * TE)
    #ut.plotim3(np.absolute(im[:,:,-10:-1]),[4,-1])

    nx,ny,nte  = im.shape
    #undersampling
    #mask       = ut.mask3d( nx, ny, nte, [15,15,0], 0.8)
    #FTm   = opts.FFT2d_kmask(mask)
    #FTm        = opts.FFTW2d_kmask(mask)
    #FTm   = opts.FFT2d()
    #b          = FTm.forward(im)
    scaling    = ut.scaling(im)
    im         = im/scaling
    ut.plotim3(np.absolute(im[:,:,:]),[4,-1],bar=1, pause_close = 2)

    #ut.plotim3(mask)
    #ut.plotim3(np.absolute(FTm.backward(b))) #undersampled imag
    #parameters
    xpar        = np.zeros((nx,ny,4), np.complex128)
    #xpar[:,:,0]  = 10*np.ones((nx,ny))
    #ut.plotim3(np.absolute(xpar),[3,-1])
    # IDEAL and FFT jointly
    IDEAL       = idealc.IDEAL_fatmyelin_opt2(TE, fat_freq_arr , fat_rel_amp )#fat_freq_arr , fat_rel_amp
    Aideal_ftm  = IDEAL#opts.joint2operators(IDEAL, FTm)#(FTm,IDEAL)#
    IDEAL.set_x(xpar) #should update in each gauss newton iteration
    residual    = IDEAL.residual(im)
    #ut.plotim3(np.absolute(FTm.backward(residual)))
    # wavelet and x+d_x
    addx_water   = idealc.x_add_dx()
    addx_fat     = idealc.x_add_dx()
    addx_dfwater = idealc.x_add_dx()
    addx_dffat   = idealc.x_add_dx()

    #addx        = idealc.x_add_dx()
    #addx.set_x(xpar)
    #addx.set_w([1, 1, 0.0001])
    addx_water.set_x   (xpar[...,0]) #should update in each gauss newton iteration
    addx_fat.set_x     (xpar[...,1])
    addx_dfwater.set_x (xpar[...,2])
    addx_dffat.set_x   (xpar[...,3])

    dwt          = opts.DWT2d(wavelet = 'haar', level=4)
    tvop         = tvopc.TV2d_r()
    Adwt_addx_w  = opts.joint2operators(tvop, addx_water)   
    Adwt_addx_f  = opts.joint2operators(tvop, addx_fat)
    Adwt_addx_dwat = opts.joint2operators(tvop, addx_dfwater)  
    Adwt_addx_dfat = opts.joint2operators(tvop, addx_dffat)  

    #Adwt_addx   = opts.joint2operators(dwt, addx)

    #CGD
    Nite  = 100
    l1_r1 = 0.01
    l1_r2 = 0.01
    l1_r3 = 0.01
    l1_r4 = 0.01
    def f(xi):
        #return np.linalg.norm(Aideal_ftm.forward(xi)-residual)
        return alg.obj_fidelity(Aideal_ftm, xi, residual) #\
        + l1_r1 * alg.obj_sparsity(Adwt_addx_w, xi[...,0])\
        + l1_r2 * alg.obj_sparsity(Adwt_addx_f, xi[...,1])\
        + l1_r3 * alg.obj_sparsity(Adwt_addx_dwat, xi[...,2])\
        + l1_r4 * alg.obj_sparsity(Adwt_addx_dfat, xi[...,3])

    def df(xi):
        #return 2*Aideal_ftm.backward(Aideal_ftm.forward(xi)-residual)
        gradall = alg.grad_fidelity(Aideal_ftm, xi, residual)
        gradall[...,0] += l1_r1 * alg.grad_sparsity(Adwt_addx_w, xi[...,0])
        gradall[...,1] += l1_r2 * alg.grad_sparsity(Adwt_addx_f, xi[...,1])
        gradall[...,2] += l1_r3 * alg.grad_sparsity(Adwt_addx_dwat, xi[...,2]) 
        gradall[...,3] += l1_r4 * alg.grad_sparsity(Adwt_addx_dfat, xi[...,3]) 

        return gradall

    #do soft thresholding
    #Nite = 200 #number of iterations
    #step = 0.1 #step size
    #th   = 0.001 # theshold level
    #do tv cs mri recon
    #Nite = 20 #number of iterations
    #step = 1 #step size
    #tv_r = 0.01 # regularization term for tv term
    #rho  = 1.0  
    #ostep = 0.3       
    for i in range(40):
        #wavelet L1 IST
    #    dxpar = solvers.IST_3( Aideal_ftm.forward, Aideal_ftm.backward,\
    #                Adwt_addx.backward, Adwt_addx.forward, residual, Nite, step, th )
        #wavelet L1 ADMM
    #    dxpar = solvers.ADMM_l2Afxnb_l1Tfx( Aideal_ftm.forward, Aideal_ftm.backward, \
    #               Adwt_addx.backward, Adwt_addx.forward, residual, Nite, step, tv_r, rho,25 )
        # TV ADMM
    #    dxpar = solvers.ADMM_l2Afxnb_tvx( Aideal_ftm.forward, Aideal_ftm.backward, residual\
    #    	, Nite, step, tv_r, rho ) 
        # L2 CGD
    #    dxpar = pf.prox_l2_Afxnb_CGD2( Aideal_ftm.forward, Aideal_ftm.backward, residual, rho, Nite )
    #    dxpar = pf.prox_l2_Afxnb_CGD2( Aideal_ftm.forward, Aideal_ftm.backward, residual, Nite )
        # L1 CGD
        #dxpar = pf.prox_l2_Afxnb_CGD2( IDEAL.forward, IDEAL.backward, residual, Nite )
        dxpar   = alg.conjugate_gradient(f, df, Aideal_ftm.backward(residual), Nite )
        ostep,j = alg.BacktrackingLineSearch(f, df, xpar, dxpar)
        if i%1 == 0:
            nxpar = xpar + ostep*dxpar
            nxpar[...,1] = 10*nxpar[...,1]
            ut.plotim3(np.absolute(nxpar)[...,0:2],colormap='viridis', bar=1, vmin = 0, vmax = 1, pause_close = 2)
            ut.plotim3(b0_gain * np.real(nxpar)[...,2],colormap='viridis',bar=1, pause_close = 2)
            ut.plotim3(b0_gain * np.imag(nxpar)[...,2],colormap='viridis',bar=1, pause_close = 2)
            ut.plotim3(b0_gain * np.real(nxpar)[...,3],colormap='viridis',bar=1, pause_close = 2)
            ut.plotim3(b0_gain * np.imag(nxpar)[...,3],colormap='viridis',bar=1, pause_close = 2)
            sio.savemat(datpath + 'cs_ideal_fitting/cs_IDEAL_CGD.mat', {'xpar': xpar, 'residual': residual})
        xpar = xpar + ostep*dxpar#.astype(np.float64)   

        #if i > 1: #fix the frequence offset to be equal for two components
        #    freq_ave    = 0.5 * np.real(xpar[:,:,2]) + 0.5 * np.real(xpar[:,:,3])
        #    xpar[:,:,2] = freq_ave +1j*(np.imag(xpar[:,:,2]))
        #    xpar[:,:,3] = freq_ave +1j*(np.imag(xpar[:,:,3]))

        IDEAL.set_x(xpar) #should update in each gauss newton iteration
        residual = IDEAL.residual(im)
        ut.plotim3(np.absolute(residual),[4,-1],bar=1, pause_close = 2)

        sio.savemat('../save_data/myelin/ideal_result_cg.mat', \
             {'xpar':xpar, 'residual':residual})
        
        #addx.set_x(xpar) #should update in each gauss newton iteration
        addx_water.set_x(xpar[...,0]) #should update in each gauss newton iteration
        addx_fat.set_x  (xpar[...,1])
        addx_dfwater.set_x(xpar[...,2])
        addx_dffat.set_x  (xpar[...,3])

    ut.plotim3(np.absolute(xpar)[...,0:2],bar=1)
    ut.plotim3(np.real(xpar + ostep*dxpar)[...,2],bar=1)
    ut.plotim3(np.imag(xpar + ostep*dxpar)[...,2],bar=1)
    ut.plotim3(np.real(xpar + ostep*dxpar)[...,3],bar=1)
    ut.plotim3(np.imag(xpar + ostep*dxpar)[...,3],bar=1)    