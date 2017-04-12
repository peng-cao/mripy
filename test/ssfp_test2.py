"""
simulate ssfp seq with fixed tr, fa and phase cycling
parrallel computation is used
usage:
execfile('sfp_test2.py')
print abs(results)
"""
from joblib import Parallel, delayed
import multiprocessing
import sim_spin as ss
import numpy as np0
import sim_seq as sseq

M0 = np.matrix([0.0,0.0,1.0]).T
tr = 10.0 #ms
fa = np.pi/2
T1 = 1000.0 #ms
T2 = 200.0 #ms
df = -40.0
PD = 1.0 #proton density
Nk = 960

inputs = range(8)
def processInput(i):
    S = sseq.sim_ssfp( M0, tr, fa, T1, T2, df, PD, Nk )
    return S

num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs)
