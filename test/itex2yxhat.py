"""
simple test of main function
usage
python itex2yxhat.py -a mrf_t1t2b0pd_mrf_randphasecyc_traintest.mat -x datax1_cs.mat -o test_out -r load_cnn_t1t2b0_1dcov.py -e /working/larson/UTE_GRE_shuffling_recon/python_test/ -d /working/larson/UTE_GRE_shuffling_recon/MRF/sim_ssfp_fa10_t1t2/IR_ssfp_t1t2b0pd7/
python x2yxhat.py -h
"""
import sys
import getopt

import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
sys.path.append('/working/larson/UTE_GRE_shuffling_recon/MRF/sim_ssfp_fa10_t1t2/IR_ssfp_t1t2b0pd7')
#from load_cnn_t1t2b0_1dcov import *
from joblib import Parallel, delayed
import multiprocessing
import sim_seq_array_data as ssad
import sim_spin as ss
import numpy as np
import sim_seq as sseq
import scipy.io as sio
import os
import tensorflow as tf
import importlib
def restorecnn( pathdat, pathexe, restorefile, inputfile2, data_x ):
    # restore tensorflow model
    # pathdat = '/working/larson/UTE_GRE_shuffling_recon/MRF/sim_ssfp_fa10_t1t2/IR_ssfp_t1t2b0pd7/'
    # pathexe = '/working/larson/UTE_GRE_shuffling_recon/python_test/'
    # restorefile = 'load_cnn_t1t2b0_1dcov.py'
    # inputfile2 = 'datax1_cs.mat'

    os.chdir(pathdat)
    print 'call restore script:',pathdat+restorefile
    #execfile(pathdat+restorefile)
    #from load_cnn_t1t2b0_1dcov import *
    cnn = importlib.import_module("load_cnn_t1t2b0_1dcov")
    # input MRF time courses
    #mat_contents2 = sio.loadmat(pathdat+inputfile2);
    #data_x2 = mat_contents2["datax1"]

    # apply CNN model, x->y
    test_outy = cnn.sess.run(cnn.y_conv, feed_dict={cnn.x: data_x, cnn.keep_prob: 1.0}); 
    
    os.chdir(pathexe)
    return test_outy 

def read_rfr_trr_mat(pathdat, inputfile1):
    # read rf and tr arrays from mat file
    # inputfile1 = 'mrf_t1t2b0pd_mrf_randphasecyc_traintest.mat' 
    mat_contents = sio.loadmat(pathdat + inputfile1);
    far = mat_contents["rf"]
    trr = mat_contents["trr"]
    return far, trr

def read_x_mat(pathdat, inputfile2):
    # input MRF time courses
    # inputfile2 = 'datax1_cs.mat'
    mat_contents2 = sio.loadmat(pathdat+inputfile2);
    data_x = mat_contents2["datax1"]
    return data_x    

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

if __name__ == "__main__":
    #intialize arguments
    outputfile = 'test_out'
    pathdat = '/working/larson/UTE_GRE_shuffling_recon/MRF/sim_ssfp_fa10_t1t2/IR_ssfp_t1t2b0pd7/'
    pathexe = '/working/larson/UTE_GRE_shuffling_recon/python_test/'
    restorefile = 'load_cnn_t1t2b0_1dcov.py'
    inputfile2 = 'datax1.mat'
    inputfile1 = 'mrf_t1t2b0pd_mrf_randphasecyc_traintest.mat' 
    argv = sys.argv
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "ha:x:o:r:e:d:", ["help","ifile1=","ifile2=","ofile=","rfile=","pathexe=","pathdat="])
        except getopt.error, msg:
             raise Usage(msg)
        #code start here
        print 'read x data from mat file, apply cnn and save y, then run simulation to generate x_hat!'
        for opt, arg in opts:
            if opt == '-h':
                print 'x2yxhat.py -a <inputfile1> -x <inputfile2> -o <outputfile> -r <restorefile> -pe <pathexe> -pd <pathdat>'
                sys.exit()
            elif opt in ("-a", "--ifile1"):
                inputfile1 = arg
            elif opt in ("-x", "--ifile2"):
                inputfile2 = arg
            elif opt in ("-o", "--ofile"):
                outputfile = arg
                print outputfile
            elif opt in ("-r", "--rfile"):
                restorefile = arg
            elif opt in ("-e", "--pathexe"):
                pathexe = arg
                print pathexe
            elif opt in ("-d", "--pathdat"):
                pathdat = arg
                print pathdat

        # read x0 from mat file
        data_x0 = read_x_mat(pathdat, inputfile2)

        # prepare for sequence simulation, y->x_hat
        far, trr = read_rfr_trr_mat(pathdat, inputfile1)
        Nk = far.shape[1]
        Nexample = data_x0.shape[0]
        ti = 5 #ms
        M0 = np.matrix([0.0,0.0,1.0]).T 

        # x real and imag parts should be combined        
        #data_x0_c = data_x0[:,0:Nexample] + 1j*data_x0[:,-Nexample:]
        data_x0_c = data_x0[:,0:Nk]*np.exp(1j*data_x0[:,Nk:2*Nk])
        # inital x 
        data_x_acc = 1j*np.zeros( (Nexample, 2*Nk) )
        data_x_acc_c = 1j*np.zeros((Nexample,Nk))
        # step size
        step = 1.0
        # for loop start here
        for _ in range(2):
            # y from cnn!!
            test_outy = restorecnn( pathdat, pathexe, restorefile, inputfile2, data_x_acc )
            seq_data = ssad.irssfp_arrayin_data( Nexample, Nk ).set( test_outy ) #intial sequence

            # sequence simulation, y->x_hat
            seq_data = ssad.irssfp_arrayin_data( Nexample, Nk )
            # parallel computing all time courses to generate x_hat
            inputs = range(Nexample)
            def processFunc(i):
                S = seq_data.sim_seq_tc(i,M0, trr, far, ti )
                return S
            Njobs = 16 #number of parallel tasks
            num_cores = multiprocessing.cpu_count()
            test_outxhat = Parallel(n_jobs=Njobs, verbose=5)(delayed(processFunc)(i) for i in inputs)

            # x_acc += step*(x0-x), later should use the A^H A (x0-x)
            data_x_acc_c = data_x_acc_c + step*(data_x0_c-test_outxhat)

            #seperate real/imag parts or abs/angle parts
            data_x_acc[:,0:Nk] = np.absolute(data_x_acc_c)
            data_x_acc[:,Nk:2*Nk] = np.angle(data_x_acc_c)  
        # for loop stop here
 
        #save y and x_hat
        sio.savemat(pathdat+ outputfile + 'y.mat', {'test_outy': test_outy})#outputfile = 'test_out'
        #sio.savemat(pathdat+outputfile+'xhat.mat', {'test_outxhat': test_outxhat})

        #code end here
    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        sys.exit(2)
