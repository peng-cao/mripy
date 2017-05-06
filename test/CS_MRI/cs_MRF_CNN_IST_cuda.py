"""
simulate mrf irssfp seq with in arrays of tr and fa
cuda parrallel computation is used

for cnn model, the input is x (raw mrf data, i) and output is y (parameters, p  )
for bloch sim, the input is y (parameters, p  ) and output is x (raw mrf data, i)

in this code, we minimize the cost funciton 
||M*f(p)-b||_2^2 + ||p||_1
using IST
where f(p) is cnn model, f'(i) is bloch sim

usage:
python test.py
# in test.py
import test.CS_MRI.cs_MRF_CNN_IST_cuda as cs_MRF_CNN_IST_cuda
cs_MRF_CNN_IST_cuda.test()
"""
import numpy as np
import scipy.io as sio
import os
import pics.proximal_func as pf
import utilities.utilities_func as ut
from numba import cuda
import numba
from math import cos, sin, exp
from numpy.linalg import solve
from cmath import phase
import utilities.utilities_class as utc
import bloch_sim.sim_seq_MRF_irssfp_cuda as ssmrf
import pics.operators_class as opts

# restore tensorflow model
pathdat = '/working/larson/UTE_GRE_shuffling_recon/MRF/sim_ssfp_fa10_t1t2/IR_ssfp_t1t2b0pd5/'
pathexe = '/home/pcao/git/mripy/'
os.chdir(pathdat)
# tensorflow release some unused gpu memory
execfile('load_cnn_t1t2b0_1dcov_gpumomgrow.py')#restore model
os.chdir(pathexe)

import pics.operators_cuda_class as cuopts


def tv_par( nx, ny, test_outy ):
    # tv regularization
    tmp = np.zeros((nx,ny,4), dtype = test_outy.dtype)
    for i in range(4):
        tmp[:,:,i] = pf.prox_tv2d( test_outy.reshape((nx,ny,4),order='F')[:,:,i], 0.04, step = 0.1 )            
        #ut.plotim1((tmp[:,:,i]))
    test_outy = tmp.reshape((nx*ny,4),order='F')
    return test_outy

def l0_par( nx, ny, test_outy ):
    # tv regularization
    tmp = np.zeros((nx,ny,4), dtype = test_outy.dtype)
    for i in range(4):
        if i is 3:#apply tv penalty on b0 map
            tmp[:,:,i] = pf.prox_l0_hard_thresh( test_outy.reshape((nx,ny,4),order='F')[:,:,i], 0.01)
        else: #apply l1 penalty on t1, t2, and pd maps
            tmp[:,:,i] = test_outy.reshape((nx,ny,4),order='F')[:,:,i]
    test_outy = tmp.reshape((nx*ny,4),order='F')
    return test_outy

def wavel1_par( nx, ny, test_outy ):
    dwt = opts.DWT2d(wavelet = 'haar', level=4)
    # tv regularization
    tmp = np.zeros((nx,ny,4), dtype = test_outy.dtype)
    for i in range(4):
        tmp[:,:,i] = pf.prox_l1_Tf_soft_thresh2( dwt.backward, dwt.forward, \
            test_outy.reshape((nx,ny,4),order='F')[:,:,i], 0.1)
        #ut.plotim1((tmp[:,:,i]))
    test_outy = tmp.reshape((nx*ny,4),order='F')
    return test_outy

def constraints( test_outy, th = 0.001 ):
    test_outy[test_outy < th]  = 0.0
    test_outy[test_outy > 1.0] = 1.0    
    for i in range(test_outy.shape[0]):
        if test_outy[i,3] < 0.2:
            test_outy[i,:] = np.array([0.0,0.0,0.5,0.0])#0.1*test_outy[i,:]#np.zeros(4, np.float64)
    return test_outy

def mask_ksp3d( nx, ny, nz, FTm, x ):
    # undersampling in k-space
    mtx = x.reshape((nx, ny, nz),order='F')   
    b = FTm.forward(mtx)
    return FTm.backward(b).reshape((nx*ny, nz),order='F').astype(x.dtype)


def test():
    # read rf and tr arrays from mat file
    mat_contents  = sio.loadmat(pathdat+'mrf_t1t2b0pd_mrf_randphasecyc_traintest.mat');
    far           = np.array(mat_contents["rf"].astype(np.complex128).squeeze())
    trr           = np.array(mat_contents["trr"].astype(np.float64).squeeze())
    # input MRF time courses
    mat_contents2 = sio.loadmat(pathdat+'datax1.mat');
    data_x        = np.array(mat_contents2["datax1"]).astype(np.float64)
    # prepare for sequence simulation, y->x_hat
    Nk            = far.shape[0]
    Nexample      = data_x.shape[0]
    ti            = 10 #ms
    M0            = np.array([0.0,0.0,1.0]).astype(np.float64)
    #image size
    nx            = 217
    ny            = 181
    # mask in ksp
    mask          = ut.mask3d( nx, ny, Nk, [15,15,0], 0.4)
    #FTm           = opts.FFT2d_kmask(mask) 
    FTm           = cuopts.FFT2d_cuda_kmask(mask)
    
    #intial timing
    timing        = utc.timing()
    # x real and imag parts should be combined
    data_x_c      = np.zeros((Nexample, Nk),np.complex128)
    data_x_c      = ssmrf.seqdata_realimag_2complex(data_x)
    #could do mask M here, data_x_c = M*data_x_c
    data_x_c      = mask_ksp3d(nx, ny, Nk, FTm, data_x_c)
    #data_x0_c     = data_x_c
    data_x        = ssmrf.seqdata_complex_2realimag(data_x_c)
    data_x0       = data_x
    # intial tmp data
    test_outy0    = np.zeros((Nexample,4),dtype=np.float64)
    test_outy     = np.zeros((Nexample,4),dtype=np.float64)
    tmptest_outy  = np.zeros((Nexample,4),dtype=np.float64)
    acctest_outy  = np.zeros((Nexample,4),dtype=np.float64)
    # f'(y), cnn model
    timing.start()
    ssmrf.batch_apply_tf_cuda( Nexample, ny, sess.run, x, data_x, y_conv, test_outy0, keep_prob ) 
    timing.stop().display('CNN model first estimation ')
    test_outy0        = constraints( test_outy0 )
    acctest_outy      = test_outy0
    # for loop start here
    for _ in range(40):
        #cuda simulate bloch for each pixel
        T1r, T2r, dfr, PDr = ssmrf.set_par((test_outy) )
        timing.start()
        data_x_c      = ssmrf.bloch_sim_batch_cuda( Nexample, 7*ny, Nk, PDr, T1r, T2r, dfr, M0, trr, far, ti )
        timing.stop().display('Bloch sim in loop ')
        #0.5* d||M*f(p)-b||_2^2/dp = f'*(M*f(p)-b) = f'(M*f(p)) - f'(b)
        # could do mask, M, here, e.g. data_x_c=M*data_x_c
        data_x_c      = mask_ksp3d(nx, ny, Nk, FTm, data_x_c)       
        #seperate real/imag parts or abs/angle parts
        data_x        = ssmrf.seqdata_complex_2realimag(data_x_c)
        # apply CNN model, f'(i)
        ssmrf.batch_apply_tf_cuda( Nexample, ny, sess.run, x, data_x, y_conv, tmptest_outy, keep_prob )
        tmptest_outy  = constraints( tmptest_outy )
        # gradient = f'(M * f(p)) - f'(b)
        tmptest_outy  = tmptest_outy - test_outy0 #+ 0.2*test_outy
        # gradient descent for test_outy
        acctest_outy  = acctest_outy - tmptest_outy
        print('gradient 0.5* d||f(p)-b||_2^2/dp: %g' % np.linalg.norm(tmptest_outy))
        #print('test_outy: %g' % np.linalg.norm(test_outy))
        acctest_outy  = constraints( acctest_outy )
        #could do soft thresholding on test_outy here, i.e. test_outy = threshold(test_outy)
        #test_outy     = l0_par(nx, ny, acctest_outy) #  
        test_outy     = tv_par(nx, ny, acctest_outy)

        sio.savemat(pathdat +'cnn_cs_testouty.mat', {'test_outy': test_outy})

if __name__ == "__main__":
    test()
