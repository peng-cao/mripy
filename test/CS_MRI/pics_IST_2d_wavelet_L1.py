"""
test soft thresholding algrithom, IST_3
"""
import numpy as np
import pics.proximal_func as pf
import pics.CS_MRI_solvers_func as solvers
import pics.operators_class as opts
import utilities.utilities_func as ut
from espirit.espirit_func import espirit_2d
import scipy.io as sio
def test():
    # simulated image
    mat_contents = sio.loadmat('data/brain_32ch.mat');
    x            = mat_contents["DATA"] 
    #mask         = mat_contents["mask_randm_x3"].astype(np.float)    
    nx,ny,nc     = x.shape
    #crop k-space
    xcrop        = ut.crop2d( x, 16 )  
    if 0:#do espirit  
        Vim, sim     = espirit_2d(xcrop, x.shape,\
         nsingularv = 150, hkwin_shape = (16,16,16), pad_before_espirit = 0, pad_fact = 2 )
        #coil map
        ut.plotim3(np.absolute(Vim),[4,-1],bar = 1)
        ut.plotim1(np.absolute(sim),bar = 1)
        #create espirit operator
        esp = opts.espirit(Vim)
        esp.save('../save_data/espirit_data_2d.mat')
    else:
        esp = opts.espirit()
        esp.restore('../save_data/espirit_data_2d.mat')
    #create mask
    mask = ut.mask2d( nx, ny, center_r = 15, undersampling = 0.25 )
    FTm  = opts.FFT2d_kmask(mask)
    ut.plotim1(np.absolute(mask))#plot the mask
    Aopt = opts.joint2operators(esp, FTm)
    #create image
    im   = FTm.backward(x)
    #ut.plotim3(np.absolute(im[:,:,:]))
    #wavelet operator
    dwt  = opts.DWT2d(wavelet = 'haar', level=4)
    # undersampling in k-space
    b = FTm.forward(im)
    scaling = ut.optscaling(FTm,b)
    b = b/scaling
    ut.plotim1(np.absolute(Aopt.backward(b))) #undersampled imag
    #do soft thresholding
    Nite = 50 #number of iterations
    step = 1 #step size
    th = 0.1 # theshold level
    #xopt = solvers.IST_2(FTm.forward, FTm.backward, b, Nite, step,th)
    xopt = solvers.FIST_3( Aopt.forward, Aopt.backward, dwt.backward, dwt.forward, b, Nite, step, th )
    ut.plotim3(np.absolute(xopt))

if __name__ == "__main__":
    test()
