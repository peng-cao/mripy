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


#tv minimization
#tv_r, regularization parameter for tv term
def ADMM_l2Aguassnewton_tvx( IDEALopt, FTmopt, b, Nite, step, tv_r, rho, gn_Nite = 3, tvndim = 2 ):
    Aideal_ftm = opts.joint2operators(IDEALopt, FTmopt)#(FTm,IDEAL)#
    z = Aideal_ftm.backward(b) #np.zeros(x.shape), z=AH(b)
    u = np.zeros(z.shape, dtype = b.dtype)
    # 2d or 3d, use different proximal funcitons
    if tvndim is 2:
        tvprox = pf.prox_tv2d_r
    elif tvndim is 3:
        tvprox = pf.prox_tv3d
    else:
        print('dimension imcompatiable in ADMM_l2Afxnb_tvx')
        return None
    # iteration
    for _ in range(Nite):
        # soft threshold
        #x = pf.prox_l2_Afxnb_GD(Afunc,invAfunc,b,z-u,rho,20,0.1)
        #x = pf.prox_l2_Afxnb_CGD( Afunc, invAfunc, b, z-u, rho, gn_Nite )
        x = pf.prox_l2_gaussnewton( IDEALopt, FTmopt, b, z-u, rho, gn_Nite)
        z = tvprox(x + u, 2.0 * tv_r/rho)#pf.prox_tv2d(x+u,2*tv_r/rho)
        u = u + step * (x - z)
        print( 'gradient in ADMM %g' % np.linalg.norm(x-z))
    return x

#l1 with tranform function Tf, which can be wavelet transform
def ADMM_l2Agaussnewton_l1Tfx( IDEALopt, FTmopt, Tsparse, b, Nite, step, l1_r, rho, gn_Nite = 3 ):
    Aideal_ftm = opts.joint2operators(IDEALopt, FTmopt)#(FTm,IDEAL)#
    z = Aideal_ftm.backward(b) #np.zeros(x.shape), z=AH(b)
    u = np.zeros(z.shape, dtype = b.dtype)
    # iteration
    for _ in range(Nite):
        # soft threshold
        #x = pf.prox_l2_Afxnb_GD(Afunc,invAfunc,b,z-u,rho,10,0.1)
        #x = pf.prox_l2_Afxnb_CGD( Afunc, invAfunc, b, z-u, rho, gn_Nite )  
        x = pf.prox_l2_gaussnewton( IDEALopt, FTmopt, b, z-u, rho, gn_Nite)              
        z = pf.prox_l1_Tf_soft_thresh(Tsparse.backward,Tsparse.forward,x+u,l1_r/rho)
        u = u + step*(x-z)
        print( 'gradient in ADMM %g' % np.linalg.norm(x-z))
    return x


"""
        x1 = pf.prox_l2_Afxnb_GD(Afunc,invAfunc,b,z-u1,rho,10,0.1)
        x2 = pf.prox_l1_soft_thresh(z-u2,l1_r1/rho)
        x3 = pf.prox_l1_Tf_soft_thresh(Tfunc,invTfunc,z-u3,l1_r2/rho)
        z = (x1 + x2 + x3)/3 + (u1 + u2 + u3)/3
        u1 = u1 + step*(x1-z)
        u2 = u2 + step*(x2-z)
        u3 = u3 + step*(x3-z)
"""
#l1 with tranform function Tf, which can be wavelet transform
def ADMM_l2Agaussnewton_l1Tfx_tvx( IDEALopt, FTmopt, Tsparse, b, Nite, step, l1_r, tv_r, rho, gn_Nite = 3, tvndim = 2 ):
    Aideal_ftm = opts.joint2operators(IDEALopt, FTmopt)#(FTm,IDEAL)#
    z = Aideal_ftm.backward(b) #np.zeros(x.shape), z=AH(b)
    u1 = np.zeros(z.shape, dtype = b.dtype)
    u2 = np.zeros(z.shape, dtype = b.dtype)
    u3 = np.zeros(z.shape, dtype = b.dtype)    
    # 2d or 3d, use different proximal funcitons
    if tvndim is 2:
        tvprox = pf.prox_tv2d_r
    elif tvndim is 3:
        tvprox = pf.prox_tv3d_r
    # iteration
    for _ in range(Nite): 
        x1 = pf.prox_l2_gaussnewton( IDEALopt, FTmopt, b, z-u1, rho, gn_Nite)              
        x2 = pf.prox_l1_Tf_soft_thresh(Tsparse.backward,Tsparse.forward,z-u2,l1_r/rho)
        x3 = tvprox(z - u3, 2.0 * tv_r/rho)
        z  = (x1 + x2 + x3)/3 + (u1 + u2 + u3)/3
        u1 = u1 + step*(x1-z)
        u2 = u2 + step*(x2-z)
        u3 = u3 + step*(x3-z)
        print( 'gradient in ADMM %g' % np.linalg.norm(x1-z))
    return x1

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
 
    #ut.plotim3(np.real(im[:,:,:]))
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
    #ut.plotim3(np.absolute(FTm.backward(b))) #undersampled imag
    #parameters
    xpar        = np.zeros((nx,ny,3), np.complex128)
    #xpar[:,:,0]  = 10*np.ones((nx,ny))
    #ut.plotim3(np.absolute(xpar),[3,-1])
    # IDEAL and FFT jointly
    IDEAL       = idealc.IDEAL_opt2(TE, fat_freq_arr , fat_rel_amp )#fat_freq_arr , fat_rel_amp
    Aideal_ftm  = opts.joint2operators(IDEAL, FTm)#(FTm,IDEAL)#
    IDEAL.set_x(xpar) #this set the size of data
    dwt         = opts.DWT2d(wavelet = 'haar', level=4)

    #do tv cs mri recon
    Nite = 10 #number of iterations
    step = 1 #step size
    l1_r = 0.001
    tv_r = 0.0001 # regularization term for tv term
    rho  = 1.0   

    xpar = ADMM_l2Agaussnewton_l1Tfx(IDEAL, FTm, dwt, b, Nite, step, l1_r, rho )        
    #xpar = ADMM_l2Aguassnewton_tvx(IDEAL, FTm, b, Nite, step, tv_r, rho ) 
    #xpar = ADMM_l2Agaussnewton_l1Tfx_tvx( IDEAL, FTm, dwt, b, Nite, step, l1_r, tv_r, rho)

    ut.plotim3(np.absolute(xpar)[...,0:2],bar=1)
    ut.plotim3(b0_gain * np.real(xpar)[...,2],bar=1)
    ut.plotim3(b0_gain * 2.0 * np.pi * np.imag(xpar)[...,2],bar=1)    