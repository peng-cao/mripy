"""
test tv denoising and CS recon, algrithm using ADMM
"""
import numpy as np
import pics.CS_MRI_solvers_func as solvers
import utilities.utilities_func as ut
import pics.operators_class as opts
import scipy.io as sio


def scaling( FT, b ):
    x0 = np.absolute(FT.backward(b))
    return max(x0.flatten())

def test():
    # simulated image
    mat_contents = sio.loadmat('data/sim_3dmri.mat');
    x0 = mat_contents["sim_3dmri"]
    x = x0[:,:,40:60]
    nx,ny,nz = x.shape
    mask = ut.mask3d( nx, ny, nz, [15,15,0])


    FTm = opts.FFTnd_kmask(mask)
    #ut.plotim3(np.absolute(mask[:,:,1:10]))#plot the mask

    # undersampling in k-space
    b = FTm.forward(x)
    ut.plotim3(np.absolute(FTm.backward(b))) #undersampled imag

    scale = scaling(FTm,b)

    #b = b/scale
    
    #do cs mri recon
    Nite = 2 #number of iterations
    step = 0.5 #step size
    #th   = 1 #threshold
    #xopt = solvers.IST_2(FTm.forward,FTm.backward,b, Nite, step,1) #soft thresholding
    xopt = solvers.ADMM_l2Afxnb_tvx( FTm.forward, FTm.backward, b, Nite, step, 10, 1 )
    #xopt = solvers.ADMM_l2Afxnb_l1x_2( FTm.forward, FTm.backward, b, Nite, step, 100, 1 )

    ut.plotim3(np.absolute(xopt))

#if __name__ == "__main__":
    #test()
