"""
simulate ssfp seq with in arrays of tr and fa
parrallel computation is used
usage:
import os
pathdat = '/working/larson/UTE_GRE_shuffling_recon/MRF/sim_ssfp_fa10_t1t2/IR_ssfp_t1t2b0pd7/'
pathexe = '/working/larson/UTE_GRE_shuffling_recon/python_test/'
os.chdir(pathdat)
execfile('load_cnn_t1t2b0_1dcov.py') #restore tensorflow model
os.chdir(pathexe)

execfile('ssfp_test6.py')
print test_outxhat
"""
from joblib import Parallel, delayed
import multiprocessing
import bloch_sim.sim_seq_array_data as ssad
import bloch_sim.sim_spin as ss
import numpy as np
import bloch_sim.sim_seq as sseq
import scipy.io as sio
import os
import pics.proximal_func as pf
import utilities.utilities_func as ut

# restore tensorflow model
pathdat = '/working/larson/UTE_GRE_shuffling_recon/MRF/sim_ssfp_fa10_t1t2/IR_ssfp_t1t2b0pd5/'
pathexe = '/working/larson/UTE_GRE_shuffling_recon/python_test/'
os.chdir(pathdat)
execfile('load_cnn_t1t2b0_1dcov.py')
#execfile('load_cnn_t1t2b0_1dcov.py')#I figured out that currently I have to do this twice to get the model restured correctly
os.chdir(pathexe)

# read rf and tr arrays from mat file 
mat_contents = sio.loadmat(pathdat+'mrf_t1t2b0pd_mrf_randphasecyc_traintest.mat');
far = mat_contents["rf"]
trr = mat_contents["trr"]
# input MRF time courses
mat_contents2 = sio.loadmat(pathdat+'datax1.mat');
data_x0 = mat_contents2["datax1"]

# prepare for sequence simulation, y->x_hat
Nk = far.shape[1]
Nexample = data_x0.shape[0]
ti = 10 #ms
M0 = np.matrix([0.0,0.0,1.0]).T 

nx = 217
ny = 181

# x real and imag parts should be combined        
#data_x0_c = data_x0[:,0:Nexample] + 1j*data_x0[:,-Nexample:]
data_x0_c = data_x0[:,0:Nk]+1j*data_x0[:,Nk:2*Nk]
# inital x 
data_x_acc = data_x0 #1j*np.zeros( (Nexample, 2*Nk) )
data_x_acc_c = data_x0_c #1j*np.zeros((Nexample,Nk))

# for loop start here
for _ in range(1):
    # apply CNN model, x->y
    test_outy = sess.run(y_conv, feed_dict={x: data_x_acc, keep_prob: 1.0}); 
    #sio.savemat(pathdat+'cnn_cs_testouty.mat', {'test_outy': test_outy})

    #test_outy = np.absolute(pf.prox_l1_soft_thresh(test_outy,0.3))#l1 regularization
    # tv regularization
    tmp = 1.0*np.zeros((nx,ny,4))
    for i in range(4):
        tmp[:,:,i] = pf.prox_tv2d( test_outy.reshape((nx,ny,4),order='F')[:,:,i], 0.01, step = 0.1 )
        ut.plotgray(tmp[:,:,i])
    test_outy = tmp.reshape((nx*ny,4),order='F')

    sio.savemat(pathdat+'cnn_cs_testouty.mat', {'test_outy': test_outy})

    # intial seq simulation with t1t2b0 values
    seq_data = ssad.irssfp_arrayin_data( Nexample, Nk ).set( test_outy )

    # parallel computing all time courses
    inputs = range(Nexample)
    def processFunc(i):
        S = seq_data.sim_seq_tc(i,M0, trr, far, ti )
        return S
    Njobs = 16 #number of parallel tasks
    num_cores = multiprocessing.cpu_count()
    test_outxhat = Parallel(n_jobs=Njobs, verbose=5)(delayed(processFunc)(i) for i in inputs)
    

    # x_acc += step*(x0-x), later should use the A^H A (x0-x)
    data_x_acc_c = data_x_acc_c + 0.1*(data_x0_c-test_outxhat)
    #data_x_acc_c = test_outxhat

    #seperate real/imag parts or abs/angle parts
    data_x_acc[:,0:Nk] = np.real(data_x_acc_c)
    data_x_acc[:,Nk:2*Nk] = np.imag(data_x_acc_c) 

#save x_hat and y
#sio.savemat(pathdat+'cnn_cs_testoutxhat.mat', {'test_outxhat': test_outxhat})
sio.savemat(pathdat+'cnn_cs_testouty.mat', {'test_outy': test_outy})

