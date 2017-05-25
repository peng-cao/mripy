"""
test tv denoising and CS recon, algrithm using ADMM
"""
import numpy as np
import pics.CS_MRI_solvers_func as solvers
import utilities.utilities_func as ut
import pics.operators_class as opts
import scipy.io as sio

def test():
    # simulated image
    mat_contents = sio.loadmat('data/sim_2dmri.mat');
    x = mat_contents["sim_2dmri"]
    nx,ny = x.shape
    mask = ut.mask2d( nx, ny, center_r = 15 )
    FTm = opts.FFT2d_kmask(mask)
    ut.plotim1(np.absolute(mask))#plot the mask

    # undersampling in k-space
    b = FTm.forward(x)
    ut.plotim1(np.absolute(FTm.backward(b))) #undersampled imag

    #do cs mri recon
    Nite = 20 #number of iterations
    step = 0.5 #step size
    #th   = 1 #threshold
    #xopt = solvers.IST_2(FTm.forward,FTm.backward,b, Nite, step,th) #soft thresholding
    xopt = solvers.ADMM_l2Afxnb_tvx( FTm.forward, FTm.backward, b, Nite, step, 10, 1 )
    #xopt = solvers.ADMM_l2Afxnb_l1x_2( FTm.forward, FTm.backward, b, Nite, step, 100, 1 )

    ut.plotim1(np.absolute(xopt))

#if __name__ == "__main__":
    #test()
