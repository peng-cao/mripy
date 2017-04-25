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
from espirit.espirit_func import espirit_2d

def test():
    ft = opt.FFT2d()
    # simulated image
    mat_contents = sio.loadmat('data/brain_32ch.mat');
    x = mat_contents["DATA"]    
    #ut.plotim1(np.absolute(x[:,:,0]))
    im = ft.backward(x[:,:,:])
    ut.plotim3(np.absolute(im[:,:,:]))
    #crop k-space
    xcrop = ut.crop2d( x, 30 )  
    #do espirit2d  
    Vim, sim = espirit_2d(xcrop, x.shape,\
     nsingularv = 150, hkwin_shape = (16,16,16), pad_before_espirit = 0, pad_fact = 2 )

if __name__ == "__main__":
    test()