"""
test tv denoising and CS recon, algrithm using ADMM

usage:
python test.py
#in test.py
import test.CS_MRI.cs_TV_ADMM as cs_TV_ADMM
cs_TV_ADMM.test()
"""


# make sure you've got the following packages installed
import numpy as np
import pics.CS_MRI_solvers_func as solvers
import utilities.utilities_func as ut
import pics.operators_class as opts
import scipy.io as sio
from fft.cufft import fftnc2c_cuda, ifftnc2c_cuda



def scaling( FT, b ):
    x0 = np.absolute(FT.backward(b))
    return max(x0.flatten())

def test():
    # simulated image
    mat_contents = sio.loadmat('data/sim_3dmri.mat');
    x0 = np.array(mat_contents["sim_3dmri"]).astype(np.complex64)

    x = np.array(x0[:,:,30:68]).astype(np.complex64)
    nx,ny,nz = x.shape
    mask = ut.mask3d( nx, ny, nz, [15,15,0])

    FTm   = opts.FFTnd_kmask(mask)
    cuFTm = opts.FFTnd_cuda_kmask(mask)

    #ut.plotim3(np.absolute(mask[:,:,1:10]))#plot the mask

    # undersampling in k-space
    b = FTm.forward(x)
    ut.plotim3(np.absolute(FTm.backward(b)),[4,-1]) #undersampled imag

    scale = scaling(FTm,b)
    #b = b/scale
    #do cs mri recon
    Nite = 20 #number of iterations
    step = 0.5 #step size
    #th   = 1 #threshold
    #xopt = solvers.IST_2(cuFTm.forward,cuFTm.backward,b, Nite, step,2) #soft thresholding
    xopt = solvers.ADMM_l2Afxnb_tvx( cuFTm.forward, cuFTm.backward, b, Nite, step, 10, 5 )
    #xopt = solvers.ADMM_l2Afxnb_l1x_2( FTm.forward, FTm.backward, b, Nite, step, 100, 1 )

    ut.plotim3(np.absolute(xopt),[4, -1])

#if __name__ == "__main__":
    #test()
