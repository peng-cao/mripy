"""
No working!!!!!!!!!
test conjugate gradient algrithom, with L1 wavelet minimization and tv minimization
"""
import numpy as np
import scipy.io as sio
import pics.proximal_func as pf
import pics.CS_MRI_solvers_func as solvers
import pics.operators_class as opts
import pics.opt_alg as alg
import pics.tvop_class as tvopc
import utilities.utilities_func as ut

def test():
    # simulated image
    mat_contents = sio.loadmat('data/sim_2dmri.mat');
    im           = mat_contents["sim_2dmri"]
    #im = spimg.zoom(xorig, 0.4)
    #plotim2(im)

    #dwt = opts.DWTnd( wavelet = 'haar', level = 2, axes = (0, 1))
    dwt          = opts.DWT2d(wavelet = 'haar', level=4)
    nx,ny        = im.shape

    mask         = ut.mask2d( nx, ny, center_r = 15 )
    FTm          = opts.FFT2d_kmask(mask)
    #FTm          = opts.FFT2d()
    #ut.plotim1(np.absolute(mask))#plot the mask

    # undersampling in k-space
    b            = FTm.forward(im)
    scaling      = ut.optscaling(FTm,b)
    b            = b/scaling
    #ut.plotim1(np.absolute(FTm.backward(b))) #undersampled imag

    tvop         = tvopc.TV2d_r()
    #CGD
    Nite         = 20
    l1_r         = 0.001
    tv_r         = 0.001
    
    #def f(xi):
        #ut.plotim1(np.absolute(FTm.backward(FTm.forward(xi)-b)),bar=1)
    #    return alg.obj_fidelity(FTm, xi, b) \
    #    + l1_r * alg.obj_sparsity(dwt,  xi) \
    #    + tv_r * alg.obj_sparsity(tvop, xi)

    #def df(xi):
        #gradall  = np.zeros(xi.shape)
    #    gradall  = alg.grad_fidelity(FTm, xi, b)
    #    gradall += l1_r * alg.grad_sparsity(dwt,  xi) 
    #    gradall += tv_r * alg.grad_sparsity(tvop, xi)
    #    return gradall

    def h(xi):
        #ut.plotim1(np.absolute(FTm.backward(FTm.forward(xi)-b)),bar=1)
        return  tv_r * alg.obj_sparsity(tvop, xi) + l1_r * alg.obj_sparsity(dwt,  xi) # +

    def dh(xi):
        #gradall  = np.zeros(xi.shape, np.complex128)
        gradall  = l1_r * alg.grad_sparsity(dwt,  xi) 
        gradall += tv_r * alg.grad_sparsity(tvop, xi)
        return gradall

    #xopt   = alg.conjugate_gradient(f, df, FTm.backward(b), Nite )
    #xopt = pf.prox_l2_Afxnb_CGD2( FTm.forward, FTm.backward, b, Nite )
    xopt = pf.prox_l2_Afxnb_CGD3( FTm.forward, FTm.backward, b, h, dh, Nite )
    #do soft thresholding
    #Nite = 100 #number of iterations
    #step = 1 #step size
    #th = 1.5 # theshold level
    #xopt = solvers.IST_2(FTm.forward, FTm.backward, b, Nite, step,th)
    #xopt = solvers.IST_3( FTm.forward, FTm.backward, dwt.backward, dwt.forward, b, Nite, step, th )
    ut.plotim1(np.absolute(xopt))

if __name__ == "__main__":
    test()
