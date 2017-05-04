"""
simulate mrf irssfp seq with in arrays of tr and fa
cuda parrallel computation is used

for cnn model, the input is x (raw mrf data, i) and output is y (parameters, p  )
for bloch sim, the input is y (parameters, p  ) and output is x (raw mrf data, i)

in this code, we minimize the cost funciton 
||M*FT(x)-b||_2^2 + ||cnn(x)||_1_TV
using IST
where p = cnn(x) is cnn model, inv_cnn is bloch sim

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
execfile('load_cnn_t1t2b0_1dcov.py')#restore model
os.chdir(pathexe)


def test():
    # read rf and tr arrays from mat file
    mat_contents  = sio.loadmat(pathdat+'mrf_t1t2b0pd_mrf_randphasecyc_traintest.mat');
    far           = np.array(mat_contents["rf"].astype(np.complex128).squeeze())
    trr           = np.array(mat_contents["trr"].astype(np.float32).squeeze())

    # intial tmp data
    mat_contents  = sio.loadmat(pathdat+'test_outy.mat');
    test_outy     = np.array(mat_contents["test_outy"].astype(np.float32).squeeze())

    # input MRF time courses
    mat_contents2 = sio.loadmat(pathdat+'datax1.mat');
    data_x_acc    = np.array(mat_contents2["datax1"]).astype(np.float32)
    # prepare for sequence simulation, y->x_hat
    Nk            = far.shape[0]
    Nexample      = data_x_acc.shape[0]
    ti            = 10 #ms
    M0            = np.array([0.0,0.0,1.0]).astype(np.float32)
    #image size
    nx            = 217
    ny            = 181

    # for loop start here
    for _ in range(1):
         # apply CNN model, x-->parameters
        #test_outy = ssmrf.batch_apply_tf_cuda( Nexample, ny, sess.run, x, data_x_acc, y_conv, test_outy, keep_prob )        
        #cuda bloch simulation for each pixel, parameters-->x
        T1r, T2r, dfr, PDr = ssmrf.set_par(test_outy)
        #timing.start()
        data_x_c      = ssmrf.bloch_sim_batch_cuda( Nexample, 7*ny, Nk, PDr, T1r, T2r, dfr, M0, trr, far, ti )
        print(np.linalg.norm(data_x_c - ssmrf.seqdata_realimag_2complex(data_x_acc)))
        print(np.linalg.norm(data_x_c))
        sio.savemat(pathdat +'cnn_cs_testouty.mat', {'test_outy': test_outy})
        sio.savemat(pathdat +'cnn_cs_testouty1.mat', {'data_x_c': data_x_c})
        sio.savemat(pathdat +'cnn_cs_testouty2.mat', {'data_x_acc':  ssmrf.seqdata_realimag_2complex(data_x_acc)})        
        #data_x_acc = ssmrf.seqdata_complex_2realimag(data_x_c)

if __name__ == "__main__":
    test()
