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

execfile('ssfp_test5.py')
print test_outxhat
"""
from joblib import Parallel, delayed
import multiprocessing
import bloch_sim.sim_spin as ss
import numpy as np
import bloch_sim.sim_seq as sseq
import scipy.io as sio
import os
# restore tensorflow model
pathdat = '/working/larson/UTE_GRE_shuffling_recon/MRF/sim_ssfp_fa10_t1t2/IR_ssfp_t1t2b0pd7/'
#pathexe = '/working/larson/UTE_GRE_shuffling_recon/python_test/'
#os.chdir(pathdat)
#execfile('load_cnn_t1t2b0_1dcov.py')
#os.chdir(pathexe)

# read rf and tr arrays from mat file 
mat_contents = sio.loadmat(pathdat+'mrf_t1t2b0pd_mrf_randphasecyc_traintest.mat');
far = mat_contents["rf"]
trr = mat_contents["trr"]

mat_contents2 = sio.loadmat(pathdat+'datax1_cs.mat');
data_x2 = mat_contents2["datax1"]

test_outy = sess.run(y_conv, feed_dict={x: data_x2, keep_prob: 1.0}); 
sio.savemat(pathdat+'cnn_cs_testouty.mat', {'test_outy': test_outy})

Nk = far.shape[0]
Nexample = test_outy.shape[0]
# fixed fa and tr
#far = np.pi/10*np.ones(Nk)
#far[::2] = -1*far[::2]
#trr = 10*np.ones(Nk)

ti = 5 #ms
M0 = np.matrix([0.0,0.0,1.0]).T 

num_cores = multiprocessing.cpu_count()
Nx = 217
test_outxhat = 1j*np.zeros((Nx,Nexample/Nx,Nk))
for nx in range(Nx):
    inputs = range(Nexample/Nx)
    def processInput(i):
        idx = i + nx*(Nexample/Nx)
        T1 = 5000.0 * test_outy.item((idx,0)) #ms
        T2 = 500.0 * test_outy.item((idx,1)) #ms
        df = 100.0 * (2*test_outy.item((idx,2)) - 1) #hz
        PD = test_outy.item((idx,3)) #proton density
        if PD > 0.0001:
            S = sseq.sim_irssfp_arrayin( M0, trr, far, ti, T1, T2, df, PD, Nk )
        else:
            S = 1j*np.zeros(Nk)
        return S
    print nx
    results = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs)
    test_outxhat[nx,:,:] = results
sio.savemat(pathdat+'cnn_cs_testoutxhat.mat', {'test_outxhat': test_outxhat})

