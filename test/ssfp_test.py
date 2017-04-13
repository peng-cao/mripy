from joblib import Parallel, delayed
import multiprocessing
import bloch_sim.sim_spin as ss
import numpy as np


inputs = range(1)
def processInput(i):
    M = np.matrix([0.0,0.0,1.0]).T
    tr = 10.0 #ms
    fa = np.pi/4
    T1 = 1000.0 #ms
    T2 = 200.0 #ms
    df = -40.0
    PD = 1.0 #proton density
    Nk = 960
    S = 1j*np.zeros(Nk) #initial an array
    for k in range(Nk):
        #excitation
        M = ss.throt(fa, 0)*M
        #print M
        #free precession half TR
        A,B = ss.freeprecess(tr/2.0,T1,T2,df)
        M = A*M+PD*B.T
        #print M,A,B
        #save the MR signal
	S.itemset(k, M.item(0)+1j*M.item(1))
        #second half of TR
        M = A*M+PD*B.T
        fa = -fa;
        #print A,B
    #print i
    return S

num_cores = multiprocessing.cpu_count()
results = np.absolute(Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs))
