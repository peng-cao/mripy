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

pathdat = '/working/larson/UTE_GRE_shuffling_recon/20170718_voluteer_ir_fulksp/exp2_ir_fulksp/'

def test():
    ft = opts.FFT2d()    
    mat_contents = sio.loadmat(pathdat + 'rawdata.mat')
    x    = mat_contents["da"].squeeze(axis = 0).squeeze(axis = 3)
    mask = mat_contents["mask"].squeeze(axis = 0).squeeze(axis = 3)
    Vim  = mat_contents["calib"][40,...]
    #ut.plotim3(np.absolute(x[:,:,:,0]))
    im = ft.backward(x)
    ut.plotim3(np.absolute(im[:,:,im.shape[2]//2,:]))
    #get shape
    nx,ny,nc,nd  = x.shape
    #create espirit operator
    esp = opts.espirit(Vim)

    #FTm  = opts.FFTnd_kmask(mask)
    FTm = opts.FFTW2d_kmask(mask, threads = 5)
    #ut.plotim1(np.absolute(mask))#plot the mask
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
    #ut.plotim3(np.absolute(Aopt.backward(b))) #undersampled imag

    #do tv cs mri recon
    #Nite = 20 #number of iterations
    #step = 0.5 #step size
    #tv_r = 0.002 # regularization term for tv term
    #rho  = 1.0
    #xopt = solvers.ADMM_l2Afxnb_tvx( Aopt.forward, Aopt.backward, b, Nite, step, tv_r, rho )

    #do wavelet l1 soft thresholding
    Nite = 50 #number of iterations
    step = 1 #step size
    th   = 0.1 # theshold level
    xopt = solvers.FIST_3( Aopt.forward, Aopt.backward, dwt.backward, dwt.forward, b, Nite, step, th )
    sio.savemat(pathdat + 'mripy_recon_l1wavelet.mat', {'xopt':xopt})
    ut.plotim3(np.absolute(xopt[:,:,:]))

#if __name__ == "__main__":
    #test()
