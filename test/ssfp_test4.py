"""
simulate ssfp seq with in arrays of tr and fa
parrallel computation is used
usage:
execfile('ssfp_test4.py')
print results
"""
from joblib import Parallel, delayed
import multiprocessing
import sim_spin as ss
import numpy as np
import sim_seq as sseq
import scipy.io as sio

# read rf and tr arrays from mat file 
#mat_contents = sio.loadmat('/working/larson/UTE_GRE_shuffling_recon/MRF/sim_ssfp_fa10_t1t2/IR_ssfp_t1t2b0pd7/mrf_t1t2b0pd_mrf_randphasecyc_traintest.mat');
#far = mat_contents["rf"]
#trr = mat_contents["trr"]

Nk = 960
# fixed fa and tr
far = np.pi/10*np.ones(Nk)
far[::2] = -1*far[::2]
trr = 10*np.ones(Nk)

ti = 5 #ms
M0 = np.matrix([0.0,0.0,1.0]).T 


inputs = range(8)
def processInput(i):
    T1 = 1000.0 #ms
    T2 = 200.0 #ms
    df = 0.0
    PD = 1.0 #proton density
    S = sseq.sim_irssfp_arrayin( M0, trr, far, ti, T1, T2, df, PD, Nk )
    return S

num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs)
