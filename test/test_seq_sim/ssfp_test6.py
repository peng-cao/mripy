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

# restore tensorflow model
pathdat = '/working/larson/UTE_GRE_shuffling_recon/MRF/sim_ssfp_fa10_t1t2/IR_ssfp_t1t2b0pd7/'
pathexe = '/working/larson/UTE_GRE_shuffling_recon/python_test/'
os.chdir(pathdat)
execfile('load_cnn_t1t2b0_1dcov.py')
execfile('load_cnn_t1t2b0_1dcov.py')#I figured out that currently I have to do this twice to get the model restured correctly
os.chdir(pathexe)

# read rf and tr arrays from mat file 
mat_contents = sio.loadmat(pathdat+'mrf_t1t2b0pd_mrf_randphasecyc_traintest.mat');
far = mat_contents["rf"]
trr = mat_contents["trr"]
# input MRF time courses
mat_contents2 = sio.loadmat(pathdat+'datax1_cs.mat');
data_x2 = mat_contents2["datax1"]

# apply CNN model, x->y
test_outy = sess.run(y_conv, feed_dict={x: data_x2, keep_prob: 1.0}); 
sio.savemat(pathdat+'cnn_cs_testouty.mat', {'test_outy': test_outy})

# prepare for sequence simulation, y->x_hat
Nk = far.shape[1]
Nexample = test_outy.shape[0]
ti = 5 #ms
M0 = np.matrix([0.0,0.0,1.0]).T 

seq_data = ssad.irssfp_arrayin_data( Nexample, Nk ).set( test_outy )
# sequentially simulate all time courses
#test_outxhat = seq_data.set( test_outy ).sim_seq_all( M0, trr, far, ti )

# parallel computing all time courses
inputs = range(Nexample)
def processFunc(i):
    S = seq_data.sim_seq_tc(i,M0, trr, far, ti )
    return S
Njobs = 16 #number of parallel tasks
num_cores = multiprocessing.cpu_count()
test_outxhat = Parallel(n_jobs=Njobs, verbose=5)(delayed(processFunc)(i) for i in inputs)


#save x_hat
#sio.savemat(pathdat+'cnn_cs_testoutxhat.mat', {'test_outxhat': test_outxhat})

