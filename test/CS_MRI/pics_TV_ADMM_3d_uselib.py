"""
test tv denoising and CS recon, algrithm using ADMM
"""
import numpy as np
import pics.proximal_func as pf
import pics.CS_MRI_solvers_func as solvers
import pics.operators_class as opts
import utilities.utilities_func as ut
from espirit.espirit_func import espirit_2d, espirit_3d
import scipy.io as sio

def test():
    ft = opts.FFTnd()    
    mat_contents = sio.loadmat('/working/larson/UTE_GRE_shuffling_recon/brain_mt_recon_20160919/brain_3dMRI_32ch.mat');
    x = mat_contents["DATA"]    
    #ut.plotim3(np.absolute(x[:,:,:,0]))
    im = ft.backward(x)
    #ut.plotim3(np.absolute(im[:,:,im.shape[2]//2,:]))
    #get shape
    nx,ny,nz,nc  = x.shape
    #crop k-space
    xcrop        = ut.crop3d( x, 12 )  
    if 1:#do espirit
        Vim, sim = espirit_3d(xcrop, x.shape, 150, hkwin_shape = (12,12,12),\
         pad_before_espirit = 0, pad_fact = 2)
        #coil map
        #ut.plotim3(np.absolute(Vim[:,:,im.shape[2]//2,:]),bar = 1)
        #ut.plotim3(np.absolute(sim),bar = 1)
        #create espirit operator
        esp = opts.espirit(Vim)
        #esp.save('../save_data/espirit_data_3d.mat')
        esp.save('/working/larson/UTE_GRE_shuffling_recon/python_test/save_data/espirit_data_3d.mat') 
    else:
        esp = opts.espirit()
        #esp.restore('../save_data/espirit_data_3d.mat')
        esp.restore('/working/larson/UTE_GRE_shuffling_recon/python_test/save_data/espirit_data_3d.mat')
    #create mask
    mask = ut.mask3d( nx, ny, nz, [15,15,0])
    FTm  = opts.FFTnd_kmask(mask)
    #ut.plotim1(np.absolute(mask))#plot the mask
    Aopt = opts.joint2operators(esp, FTm)
    #create image
    im   = FTm.backward(x)
    #ut.plotim3(np.absolute(im[:,:,:]))
    #wavelet operator
    dwt  = opts.DWTnd(wavelet = 'haar', level=4)
    # undersampling in k-space
    b = FTm.forward(im)
    scaling = ut.optscaling(FTm,b)
    b = b/scaling
    #ut.plotim3(np.absolute(Aopt.backward(b))) #undersampled imag

    #do tv cs mri recon
    Nite = 20 #number of iterations
    step = 0.5 #step size
    tv_r = 0.002 # regularization term for tv term
    rho  = 1.0
    #xopt = solvers.ADMM_l2Afxnb_tvx( Aopt.forward, Aopt.backward, b, Nite, step, tv_r, rho )
    xopt = solvers.ADMM_l2Afxnb_l1Tfx( Aopt.forward, Aopt.backward, dwt.backward, dwt.forward, b, Nite, step, tv_r, rho )

    #do wavelet l1 soft thresholding
    #Nite = 50 #number of iterations
    #step = 1 #step size
    #th = 0.4 # theshold level
    #xopt = solvers.FIST_3( Aopt.forward, Aopt.backward, dwt.backward, dwt.forward, b, Nite, step, th )
  
    ut.plotim3(np.absolute(xopt[:,:,:]))

#if __name__ == "__main__":
    #test()
