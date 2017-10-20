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
import pics.operators_class as optc
import pics.operators_cuda_class as cuoptc

def test():
    path = '/home/pcao/3d_recon/'
    matfile = 'Phantom_res256_256_20.mat'
    #phantom data
    #path = '/working/larson/UTE_GRE_shuffling_recon/UTEcones_recon/20170301/scan_1_phantom/'
    #matfile = 'Phantom_utecone.mat'
    #lung data
    #path         = '/working/larson/UTE_GRE_shuffling_recon/UTEcones_recon/20170301/lung_exp4_no_prep/'
    #matfile      = 'lung_utecone.mat'
  
    mat_contents = sio.loadmat(path + matfile);

    ktraj      = mat_contents["ktraj"]
    dcf        = mat_contents["dcf"]
    kdata      = mat_contents["kdata"].astype(np.complex64)
    ncoils     = kdata.shape[3]

    #bart nufft assumes the im_shape is weighted on ktraj, so I can extract this info here
    im_shape = [2*int(np.max(ktraj[0])),2*int(np.max(ktraj[1])),2*int(np.max(ktraj[2]))]
    # remove the weighting of im_shape from ktraj
    ktraj[0,:] = ktraj[0,:]*(1.0/im_shape[0])
    ktraj[1,:] = ktraj[1,:]*(1.0/im_shape[1])
    ktraj[2,:] = ktraj[2,:]*(1.0/im_shape[2])
    #reshape the kdata, flatten the xyz dims
    kdata      = kdata.reshape((np.prod(kdata.shape[0:3]),ncoils)).squeeze()
    #call nufft3d here
    nft = cuoptc.NUFFT3d_cuda(im_shape, dcf)
    nft.normalize_set_ktraj(ktraj)
    ft  = opts.FFTnd()    

    im  = nft.backward(kdata)
    x   = ft.forward(im)
    ut.plotim3(np.absolute(im[:,:,:,1]))
    #get shape
    #nx,ny,nz,nc  = x.shape
    #crop k-space
    xcrop        = ut.crop3d( x, 12 )  
    if 1:#do espirit  
        Vim, sim = espirit_3d(xcrop, x.shape, 150, hkwin_shape = (12,12,12),\
         pad_before_espirit = 0, pad_fact = 2, sigv_th = 0.01 )
        #coil map
        #ut.plotim3(np.absolute(Vim[:,:,im.shape[2]//2,:]),bar = 1)
        #ut.plotim3(np.absolute(sim),bar = 1)
        #create espirit operator
        esp = opts.espirit(Vim)
        #esp.save('../save_data/espirit_data_3d.mat')        
        esp.save(path + 'espirit_data_3d.mat') 
    else:
        esp = opts.espirit()
        #esp.restore('../save_data/espirit_data_3d.mat')
        esp.restore(path + 'espirit_data_3d.mat')

    #ut.plotim1(np.absolute(mask))#plot the mask
    Aopt = opts.joint2operators(esp, nft)

    #wavelet operator
    #dwt  = opts.DWTnd(wavelet = 'haar', level=4)

    nkdata = Aopt.forward(im)
    print(np.linalg.norm(nkdata-kdata)/np.linalg.norm(kdata))  
    #sio.savemat(path +'test_im.mat', {'nkdata': nkdata,'kdata':kdata})

    scaling = ut.optscaling(Aopt,kdata)
    kdata = kdata/scaling

    #do tv cs mri recon
    #Nite = 20 #number of iterations
    #step = 0.5 #step size
    #tv_r = 0.002 # regularization term for tv term
    #rho  = 1.0
    #xopt = solvers.ADMM_l2Afxnb_tvx( Aopt.forward, Aopt.backward, kdata, Nite, step, tv_r, rho )

    #do wavelet l1 soft thresholding
    Nite = 20 #number of iterations
    step = 0.1 #step size
    th = 0.04 # theshold level
    xopt = solvers.IST_2( Aopt.forward, Aopt.backward, kdata, Nite, step, th )
  
    ut.plotim3(np.absolute(xopt[:,:,:,1]))

#if __name__ == "__main__":
    #test()
