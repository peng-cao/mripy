"""
test soft thresholding algrithom, IST_3
usage:
python test.py
#in test.py
import test.CS_MRI.cs_IST_2 as cs_IST_3
cs_IST_3.test()
"""


# make sure you've got the following packages installed
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize as spopt
import scipy.fftpack as spfft
import scipy.ndimage as spimg
import scipy.io as sio
import importlib
import matplotlib.cm as cm
import pics.proximal_func as pf
import pics.CS_MRI_solvers_func as solvers
import pics.operators_class as opts
import utilities.utilities_func as ut

def test():
    # simulated image
    mat_contents = sio.loadmat('data/sim_2dmri.mat');
    x = mat_contents["sim_2dmri"]
    #x = spimg.zoom(xorig, 0.4)
    #plotim2(x)

    #dwt = opts.DWTnd( wavelet = 'haar', level = 2, axes = (0, 1))
    dwt = opts.DWT2d(wavelet = 'haar', level=4)
    nx,ny = x.shape

    mask = ut.mask2d( nx, ny, center_r = 15 )
    FTm = opts.FFT2d_kmask(mask)
    #ut.plotim1(np.absolute(mask))#plot the mask

    # undersampling in k-space
    b = FTm.forward(x)
    ut.plotim1(np.absolute(FTm.backward(b))) #undersampled imag

    #do soft thresholding
    Nite = 200 #number of iterations
    step = 1 #step size
    th = 1000 # theshold level
    #xopt = solvers.IST_2(FTm.forward, FTm.backward, b, Nite, step,th)
    xopt = solvers.FIST_3( FTm.forward, FTm.backward, dwt.backward, dwt.forward, b, Nite, step, th )
    ut.plotim1(np.absolute(xopt))

if __name__ == "__main__":
    test()
