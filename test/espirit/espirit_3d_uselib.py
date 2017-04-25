import numpy as np
import scipy.io as sio
#import pics.proximal_func as pf
#import pics.CS_MRI_solvers_func as solvers
#import pics.tvop_func as tv
import utilities.utilities_func as ut
import pics.operators_class as opt
import pics.hankel_func as hk

import matplotlib.pyplot as plt
from matplotlib.mlab import PCA
from espirit.espirit_func import espirit_3d

def test():
    ft = opt.FFTnd()
    # simulated image
    mat_contents = sio.loadmat('/working/larson/UTE_GRE_shuffling_recon/brain_mt_recon_20160919/brain_3dMRI_32ch.mat');
    x = mat_contents["DATA"]    
    #ut.plotim3(np.absolute(x[:,:,:,0]))
    im = ft.backward(x)
    ut.plotim3(np.absolute(im[:,:,:,0]))
    #crop k-space
    xcrop = ut.crop3d( x, 24 )  
    #do espirit2d  
    Vim, sim = espirit_3d(xcrop, x.shape, 150, hkwin_shape = (12,12,12),\
     pad_before_espirit = 0, pad_fact = 2)
    return Vim, sim, ft

if __name__ == "__main__":
    test()