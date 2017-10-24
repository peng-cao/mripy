import config 
import fft.nufft_func as nft

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize as spopt
import scipy.fftpack as spfft
import scipy.ndimage as spimg
import scipy.io as sio
import utilities.utilities_func as ut
import utilities.utilities_class as utc
import pics.operators_class as optc

#if config.gpu_available is True:
import fft.nufft_func_cuda as nft_cuda
import pics.operators_cuda_class as cuoptc

#from numba import jit

#def plot_recon():
#    im_under = ut.loadmat('/home/pcao/3d_recon/'+'nufftrecon.mat','im_under')
#    ut.plotim3(np.absolute(im_under[:,:,0:128]),[10, -1])
#    return

"""
ute recon for one coil image
"""
def nufft_ute( ktraj, dcf, kdata, ci, im_shape ):
    #nufft for one coil
    kdata1coil = kdata[:,:,:,ci].squeeze()
    kdata1coil = np.multiply(kdata1coil, dcf)
    c = kdata1coil.flatten()
    #scale ktraj from -pi to pi
    ktraj = np.pi * ktraj/ktraj.flatten().max()#2.0 * 
    x = ktraj[0,:].flatten()
    y = ktraj[1,:].flatten()
    z = ktraj[2,:].flatten()
    #im_under = nft.nufft3d1_gaussker(x, y, z, c, im_shape[0], im_shape[1], im_shape[2], df=1.0, eps=1E-5)
    im_under = nft_cuda.nufft3d1_gaussker_cuda(x, y, z, c, im_shape[0], im_shape[1], im_shape[2], df=1.0, eps=1E-5)

    return im_under


def ncoil_nufft( ktraj, dcf, kdata, ncoils, im_shape ):
    im = np.zeros((im_shape[0], im_shape[1], im_shape[2], ncoils), dtype=kdata.dtype)
    for i in range(ncoils):#ncoils
        print('Reconstructing coil: %d/%d' % (i+1, ncoils))
        im[:,:,:,i] = nufft_ute( ktraj, dcf, kdata, i, im_shape )
    return im

def allcoil_nufft( ktraj, dcf, kdata, ncoils, im_shape ):
    im    = np.zeros((im_shape[0], im_shape[1], im_shape[2], ncoils), dtype=kdata.dtype)
    kdatashape, dcfshape = ut.dim_match(kdata.shape, dcf.shape)
    kdata = np.multiply(kdata.reshape(kdatashape), dcf.reshape(dcfshape)).squeeze()
    c     = kdata.reshape((np.prod(kdata.shape[0:2]),ncoils)).squeeze()
    ktraj = np.pi * ktraj/ktraj.flatten().max()#2.0 * 
    x     = ktraj[0,:].flatten()
    y     = ktraj[1,:].flatten()
    z     = ktraj[2,:].flatten()#*(1.0*im_shape[0]/im_shape[2])
    #im    = nft.nufft3d1_gaussker(x, y, z, c, im_shape[0], im_shape[1], im_shape[2], df=1.0, eps=1E-5)
    im    = nft_cuda.nufft3d1_gaussker_cuda(x, y, z, c, im_shape[0], im_shape[1], im_shape[2], df=1.0, eps=1E-5,  gridfast=0)
    return im

"""
calculate the root sum of square, for combining the coil elements
"""
def rss( im_coils, axis_rss=None ):
    #print(np.absolute(im_coils).flatten().max())
    #im_coils=im_coils/np.absolute(im_coils).flatten().max()
    if axis_rss is None:
        axis_rss = len(im_coils.shape) - 1 # assume the last dimention is the coil dimention
    return np.linalg.norm(im_coils, ord=None, axis = axis_rss)

def test():
    #path = '/home/pcao/3d_recon/'
    #matfile = 'Phantom_res256_256_20.mat'
    #phantom data
    path = '/working/larson/UTE_GRE_shuffling_recon/UTEcones_recon/20170301/scan_1_phantom/'
    matfile = 'Phantom_utecone.mat'
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
    nufft = cuoptc.NUFFT3d_cuda(im_shape, dcf)
    nufft.normalize_set_ktraj(ktraj)
    #nufft      = optc.NUFFT3d(im_shape, dcf)    
    #nufft.normalize_set_ktraj(ktraj)
    im_under   = nufft.backward(kdata)
    im_under   = nufft.forward_backward(im_under)
    #kdata      = nufft.forward(im_under)
    #im_under   = nufft.backward(kdata)

    #im_under = ncoil_nufft( ktraj, dcf, kdata, ncoils, im_shape )
    #im_under = allcoil_nufft( ktraj, dcf, kdata, ncoils, im_shape )

    sio.savemat(path +'nufftrecon2.mat', {'im_under': im_under})
    #im_under = ut.loadmat(path +'nufftrecon.mat','im_under')
    rss_im_under = rss(im_under)
    ut.plotim3(np.absolute(im_under[:,:,im_shape[2]//2,:].squeeze()),[8,-1])
    ut.plotim3(rss_im_under)
    sio.savemat(path+'nufftrecon_rss2.mat', {'rss_im_under': rss_im_under})
#if __name__ == "__main__":
    #test()
