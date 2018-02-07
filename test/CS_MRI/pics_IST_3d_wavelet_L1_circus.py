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
import h5py
import pics.operators_cuda_class as cuopts

pathdat = '/working/larson/UTE_GRE_shuffling_recon/circus_firstdata_t1t2/kspace_data_vps50_13TIs/'
#pathdat = '/working/larson/UTE_GRE_shuffling_recon/circus_20180119_invivo/pw2ms/scan1/'

def test1():
    #ft = opts.FFTnd()    
    #mat_contents = sio.loadmat(pathdat + 'rawdata2.mat');
    #x = mat_contents["dataall"][...,0,11:13].astype(np.complex64)#.squeeze(axis = 4)  
    #Vim = mat_contents["calib"].astype(np.complex64)
    #mask = mat_contents["maskn"][...,0,11:13].astype(np.complex64)#.squeeze(axis = 4)   

    mat_contents = h5py.File(pathdat+'rawdata.mat')


    x          = mat_contents['dataall'][:].transpose([5,4,3,2,1,0]).squeeze(axis=4).view(np.complex128).astype(np.complex64)
    Vim        = mat_contents["calib"][:].transpose([3,2,1,0]).view(np.complex128).astype(np.complex64)
    mask       = mat_contents["maskn"][:].transpose([5,4,3,2,1,0]).squeeze(axis=4)

    for dd in range(x.shape[-1]):
        esp = opts.espirit(Vim)
        #FTm  = opts.FFTnd_kmask(mask...,dd])
        #FTm = opts.FFTWnd_kmask(mask[...,dd], threads = 15)
        FTm = cuopts.FFTnd_cuda_kmask(mask[...,dd])
        Aopt = opts.joint2operators(esp, FTm)
        #wavelet operator
        dwt  = opts.DWTnd(wavelet = 'db2', level=4, axes = (0, 1, 2))
        # undersampling in k-space
        b = x[...,dd]
        scaling = ut.optscaling(Aopt,b)
        b = b/scaling
        #do wavelet l1 soft thresholding
        xopt = scaling*solvers.FIST_3( Aopt.forward, Aopt.backward, dwt.backward, dwt.forward, b, Nite=25, step=0.5, th=0.1 )
        sio.savemat(pathdat + 'mripy_recon_l1wavelet'+ str(dd) +'.mat', {'xopt':xopt})
        #print(xopt.shape)
        #ut.plotim3(np.absolute(xopt[...,0].squeeze()))

def test():
    #ft = opts.FFTnd()    
    #mat_contents = sio.loadmat(pathdat + 'rawdata2.mat');
    #x = mat_contents["dataall"][...,0,11:13].astype(np.complex64)#.squeeze(axis = 4)  
    #Vim = mat_contents["calib"].astype(np.complex64)
    #mask = mat_contents["maskn"][...,0,11:13].astype(np.complex64)#.squeeze(axis = 4)   

    mat_contents = h5py.File(pathdat+'rawdata.mat')


    x          = mat_contents['dataall'][:].transpose([5,4,3,2,1,0]).squeeze(axis=4).view(np.complex64).astype(np.complex64)
    Vim        = mat_contents["calib"][:].transpose([3,2,1,0]).view(np.complex128).astype(np.complex64)
    mask       = mat_contents["maskn"][:].transpose([5,4,3,2,1,0]).squeeze(axis=4)

    esp = opts.espirit(Vim)
    #FTm  = opts.FFTnd_kmask(mask)
    #FTm = opts.FFTWnd_kmask(mask, threads = 15)
    FTm = cuopts.FFTnd_cuda_kmask(mask)
    Aopt = opts.joint2operators(esp, FTm)
    #wavelet operator
    dwt  = opts.DWTnd(wavelet = 'db2', level=2, axes = (0, 1, 2))
    # undersampling in k-space
    scaling = ut.optscaling(Aopt,x)
    x = x/scaling
    #do wavelet l1 soft thresholding
    xopt = scaling*solvers.FIST_3( Aopt.forward, Aopt.backward, dwt.backward, dwt.forward, x, Nite=25, step=0.5, th=0.2)
    sio.savemat(pathdat + 'mripy_recon_l1wavelet.mat', {'xopt':xopt})
    #print(xopt.shape)
    #ut.plotim3(np.absolute(xopt[...,0].squeeze()))


#if __name__ == "__main__":
    #test()
