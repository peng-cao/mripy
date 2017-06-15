"""
test soft thresholding algrithom, IST with L1 wavelet minimization
"""
import numpy as np
import scipy.io as sio
import pics.proximal_func as pf
import pics.CS_MRI_solvers_func as solvers
import pics.operators_class as opts
import utilities.utilities_func as ut

def test():
    # simulated image
    mat_contents = sio.loadmat('data/sim_2dmri.mat');
    im = mat_contents["sim_2dmri"]
    #im = spimg.zoom(xorig, 0.4)
    #plotim2(im)

    #dwt = opts.DWTnd( wavelet = 'haar', level = 2, axes = (0, 1))
    dwt = opts.DWT2d(wavelet = 'haar', level=4)
    nx,ny = im.shape

    mask = ut.mask2d( nx, ny, center_r = 15 )
    FTm = opts.FFT2d_kmask(mask)
    ut.plotim1(np.absolute(mask))#plot the mask

    # undersampling in k-space
    b = FTm.forward(im)
    scaling = ut.optscaling(FTm,b)
    b = b/scaling
    ut.plotim1(np.absolute(FTm.backward(b))) #undersampled imag

    #do soft thresholding
    Nite = 100 #number of iterations
    step = 1 #step size
    th = 1.5 # theshold level
    #xopt = solvers.IST_2(FTm.forward, FTm.backward, b, Nite, step,th)
    xopt = solvers.IST_3( FTm.forward, FTm.backward, dwt.backward, dwt.forward, b, Nite, step, th )
    ut.plotim1(np.absolute(xopt))

if __name__ == "__main__":
    test()
