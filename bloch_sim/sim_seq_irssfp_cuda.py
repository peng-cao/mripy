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
#from joblib import Parallel, delayed
#import multiprocessing
#import bloch_sim.sim_seq_array_data as ssad
#import bloch_sim.sim_spin as ss
import numpy as np
#from sim_seq import sim_irssfp_arrayin
import scipy.io as sio
import os
#import proximal_func as pf
import utilities.utilities_func as ut
from numba import cuda
import numba
#import sim_spin as ss
from math import cos, sin, exp
from numpy.linalg import solve
from cmath import phase 
#from time import time
import sim_spin_cuda as ss_cu
import utilities.utilities_class as utc

def set_par( t1t2dfpdr ):
    T1 = 5000.0 * np.array(t1t2dfpdr[:,0].squeeze()) #ms
    T2 = 500.0 * np.array(t1t2dfpdr[:,1].squeeze()) #ms
    df = 100.0 * np.array((2*t1t2dfpdr[:,2].squeeze() - 1)) #hz
    PD = np.array(t1t2dfpdr[:,3].squeeze()) #proton density
    return T1, T2, df, PD

@cuda.jit
def bloch_sim_irssfp_cuda( Nexample, Nk, PDr, T1r, T2r, dfr, M0, trr, far, ti, S ):
    i  = cuda.grid(1)
    if i > Nexample:  
        return  
    # claim local memory
    Rz   = cuda.local.array(shape=(3, 3), dtype=numba.float32)
    Rx   = cuda.local.array(shape=(3, 3), dtype=numba.float32)
    Mtmp = cuda.local.array(shape=3,      dtype=numba.float32)
    M    = cuda.local.array(shape=3,      dtype=numba.float32)    
    Rth  = cuda.local.array(shape=(3, 3), dtype=numba.float32)
    Rtho = cuda.local.array(shape=(3, 3), dtype=numba.float32)
    Em   = cuda.local.array(shape=(3, 3), dtype=numba.float32)
    Afp  = cuda.local.array(shape=(3, 3), dtype=numba.float32)
    Bfp  = cuda.local.array(shape=3,      dtype=numba.float32) 
    # sequence has multiple BSSFP-TRs for one IR
    if PDr[i] > 0.0001 and T1r[i] > 0.0001 and T2r[i] > 0.0001:
        PD = PDr[i] # proton density
        T1 = T1r[i] # T1
        T2 = T2r[i] # T2
        df = dfr[i] # freq offset
        # M0=[0 0 1] is proton density weighted
        ss_cu.veccopy_cuda(M, M0)
        M = ss_cu.vmuls_cuda(M,PD)
        #inversion pulse, phi = np.pi
        ss_cu.excitation_cuda( Mtmp, M, Rtho, Rz, Rx, Rth, np.pi, 0. )
        ss_cu.veccopy_cuda(M, Mtmp)
        #relaxation during inversion time
        ss_cu.relaxation_cuda( Mtmp, M, Afp, Bfp, Rz, Em, ti, T1, T2, df, PD )
        ss_cu.veccopy_cuda(M, Mtmp)
        #loop for TRs within one IR segment
        for k in range(Nk):
            fa = far[k] # rf complex
            fa_angle = phase(fa)  # rf phase 
            fa_absolute = abs(fa) # rf amplitude, flip angle
            tr = trr[k]
            #excitation
            ss_cu.excitation_cuda( Mtmp, M, Rtho, Rz, Rx, Rth, fa_absolute, fa_angle )
            ss_cu.veccopy_cuda(M, Mtmp)
            #free precession half TR    
            ss_cu.relaxation_cuda( Mtmp, M, Afp, Bfp, Rz, Em, tr/2.0, T1, T2, df, PD )
            ss_cu.veccopy_cuda(M, Mtmp)
            #save the MR signal on the spin echo center
            S[i, k]=M[0]+1j*M[1]
            #second half of TR
            ss_cu.relaxation_cuda( Mtmp, M, Afp, Bfp, Rz, Em, tr/2.0, T1, T2, df, PD )
            ss_cu.veccopy_cuda(M, Mtmp)
    else:
        S[i, k]=1j*0.

# do griding with cuda acceleration
def bloch_sim_batch_cuda( Nexample, batch_size, Nk, PDr, T1r, T2r, dfr, M0, trr, far, ti ):
    sim_out    = np. zeros((Nexample, Nk),   dtype = np.complex128)   #final output data
    batch_data = np. zeros((batch_size, Nk), dtype = np.complex128)   #batch output data
    bT1r       = np. zeros(batch_size,       dtype = np.float32)      #batch T1, T2, PD, df arrays
    bT2r       = np. zeros(batch_size,       dtype = np.float32)
    bPDr       = np. zeros(batch_size,       dtype = np.float32)
    bdfr       = np. zeros(batch_size,       dtype = np.float32)
    #set total number of threads on GPU
    device = cuda.get_current_device()
    tpb = device.WARP_SIZE
    bpg = int(np.ceil(float(batch_size)/tpb))
    # batch loop
    for nb in range(Nexample//batch_size):
        #print('Doing batch %d/%d for applying Bloch sim' % (nb+1,Nexample//batch_size))
        bstart = nb*batch_size      #batch start index
        bstop = bstart + batch_size #batch stop inex
        #print('%d:%d' % (bstart,bstop))
        bT1r = T1r[bstart:bstop]    #batch T1, T2, PD, df arrays
        bT2r = T2r[bstart:bstop]   
        bPDr = PDr[bstart:bstop]
        bdfr = dfr[bstart:bstop]
        #start the parallel computing on GPU
        bloch_sim_irssfp_cuda[bpg, tpb]( batch_size, Nk, bPDr, bT1r, bT2r, bdfr, M0, trr, far, ti, batch_data )
        sim_out[bstart:bstop,:] = batch_data
    return sim_out

# apply CNN model, x->y, 
def batch_apply_tf_cuda( Nexample, batch_size, sess_run, x, data_x_acc, y_conv, test_outy ):
    for nb in range(Nexample//batch_size):
        #print('Doing batch %d/%d for applying CNN' % (nb+1,Nexample//batch_size))
        bstart = nb*batch_size      #batch start index
        bstop = bstart + batch_size #batch stop inex
        test_outy[bstart:bstop,:] = sess.run(y_conv, feed_dict={x: data_x_acc[bstart:bstop,:], keep_prob: 1.0}); 
    return test_outy




if __name__ == "__main__":
    # restore tensorflow model
    pathdat = '/working/larson/UTE_GRE_shuffling_recon/MRF/sim_ssfp_fa10_t1t2/IR_ssfp_t1t2b0pd5/'
    pathexe = '/working/larson/UTE_GRE_shuffling_recon/python_test/'
    os.chdir(pathdat)
    execfile('load_cnn_t1t2b0_1dcov.py')
    #execfile('load_cnn_t1t2b0_1dcov.py')#I figured out that currently I have to do this twice to get the model restured correctly
    os.chdir(pathexe)

    # read rf and tr arrays from mat file 
    mat_contents = sio.loadmat(pathdat+'mrf_t1t2b0pd_mrf_randphasecyc_traintest.mat');
    far = np.array(mat_contents["rf"].astype(np.complex128).squeeze())
    trr = np.array(mat_contents["trr"].astype(np.int64).squeeze())

    # input MRF time courses
    mat_contents2 = sio.loadmat(pathdat+'datax1.mat');
    data_x0 = mat_contents2["datax1"]

    # prepare for sequence simulation, y->x_hat
    Nk = far.shape[0]
    Nexample = data_x0.shape[0]
    ti = 10 #ms
    M0 = np.array([0.0,0.0,1.0]).astype(np.float32) 

    nx = 217
    ny = 181

    timing = utc.timing()
    #times = []
    # x real and imag parts should be combined        
    #data_x0_c = data_x0[:,0:Nexample] + 1j*data_x0[:,-Nexample:]
    data_x0_c = data_x0[:,0:Nk]+1j*data_x0[:,Nk:2*Nk]
    # inital x 
    data_x_acc = data_x0 #1j*np.zeros( (Nexample, 2*Nk) )
    data_x_acc_c = 1j*np.zeros((Nexample,Nk))#data_x0_c

    mat_contents2 = sio.loadmat(pathdat+'cnn_cs_testouty.mat'); 
    test_outy0    = mat_contents2["test_outy"].astype(np.float32)
    test_outy     = np.zeros((Nexample,4),dtype=np.float32)
    dtest_outy    = np.zeros((Nexample,4),dtype=np.float32)
    # for loop start here
    for _ in range(10):
        #cuda simulate bloch for each pixel
        T1r, T2r, dfr, PDr = set_par( test_outy )
        timing.start()
        test_outxhat = bloch_sim_batch_cuda( Nexample, 7*ny, Nk, PDr, T1r, T2r, dfr, M0, trr, far, ti )
        timing.stop().display() 

        #d||f(x)-y||_2^2/dx = 2*f'(x)*(f(x)-y)
        #f(x) - y
        data_x_acc_c = 2*(test_outxhat-data_x0_c)

        #seperate real/imag parts or abs/angle parts
        data_x_acc[:,0:Nk] = np.real(data_x_acc_c)
        data_x_acc[:,Nk:2*Nk] = np.imag(data_x_acc_c)

        # apply CNN model, x->y, data_x_acc->CNN->test_outy 
        # 2*f'(x)*(f(x)-y)
        batch_apply_tf_cuda( Nexample, ny, sess.run, x, data_x_acc, y_conv, test_outy )
        # d||x-x0||_2^2/dx = 2* -x0 * (x-x0)
        dtest_outy = test_outy -0.1*test_outy0

        test_outy = test_outy - dtest_outy
        sio.savemat(pathdat+'cnn_cs_testouty.mat', {'test_outy': test_outy})
