"""
function simulate MRF and perform the training of cnn model
"""

from joblib import Parallel, delayed
import bloch_sim.sim_seq_array_data as ssad
import bloch_sim.sim_spin as ss
import numpy as np
import bloch_sim.sim_seq as sseq
import scipy.io as sio
import os
import tensorflow as tf
import bloch_sim.sim_seq_MRF_irssfp_cuda as ssmrf

pathdat  = '/working/larson/UTE_GRE_shuffling_recon/MRF/sim_ssfp_fa10_t1t2/IR_ssfp_t1t2b0pd5/'
pathexe  = '/home/pcao/git/mripy/test/neural_network_training/'
pathsave = '/home/pcao/git/nn_checkpoint/'

import neural_network.tf_wrap as tf_wrap
from neural_network.tf_layer import tf_layer

# these functions should be defined specifically for individal neural network
# example of the prediction function, defined using tensorflow lib
def tf_prediction_func( model ):
    #if model.arg is None:
    #    model.arg = [1.0, 1.0]
    # get data size
    NNlayer     = tf_layer()
    data_size   = int(model.data.get_shape()[1])
    target_size = int(model.target.get_shape()[1])
    mid_size    = 256*5
    # one full connection layer    
    #y1 = NNlayer.full_connection(model.data, in_fc_wide = data_size, out_fc_wide = mid_size,    activate_type = 'ReLU')
    #y  = NNlayer.full_connection(y1,         in_fc_wide = mid_size,  out_fc_wide = target_size, activate_type = 'ReLU')
    y   = NNlayer.multi_full_connection(model.data, n_fc_layers = 6, \
                                        in_fc_wide_arr = (data_size,    data_size//2, data_size//4, data_size//8,  data_size//16, data_size//32),\
                                       out_fc_wide_arr = (data_size//2, data_size//4, data_size//8, data_size//16, data_size//32, target_size),\
                                         activate_type = 'ReLU')
    # softmax output
    return tf.nn.sigmoid(y)

# example of the prediction function, defined using tensorflow lib
def tf_optimize_func( model ):
    #model.arg = [0.5, 0.5]
    loss = tf.reduce_sum(tf.pow(tf.subtract(model.prediction, model.target),2))    
    optimizer = tf.train.RMSPropOptimizer(1e-4)
    # minimization apply to cross_entropy
    return optimizer.minimize(loss)

# example of the error function, defined using tensorflow lib
def tf_error_func( model ):
    #model.arg = [1.0, 1.0]
    #training accuracy
    correct_prediction = tf.pow(tf.subtract(model.prediction, model.target),2)
    return tf.reduce_mean(correct_prediction) 


#############################

def test():
    Nk            = 960#far.shape[0]    
    model = tf_wrap.tf_model_top([None,  5*Nk], [None, 4], tf_prediction_func, tf_optimize_func, tf_error_func)

    batch_size = 800    
    # load far and trr
    # read rf and tr arrays from mat file
    #mat_contents  = sio.loadmat(pathdat+'mrf_rf_tr.mat');
    #far           = np.array(mat_contents["rf"].astype(np.complex128).squeeze())
    #trr           = np.array(mat_contents["trr"].astype(np.float64).squeeze())
    # prepare for sequence simulation, y->x_hat
    ti            = 10 #ms
    M0            = np.array([0.0,0.0,1.0]).astype(np.float64) 

    #run tensorflow on cpu, count of gpu = 0
    config = tf.ConfigProto()#(device_count = {'GPU': 0})
    #allow tensorflow release gpu memory
    config.gpu_options.allow_growth=True

    #run for 2000
    for i in range(2000):
        batch_ys           = np.random.uniform(0,1,(batch_size,4)).astype(np.float64)
        #batch_ys[:,2] = np.zeros(batch_size)
        batch_ys[:,3]      = np.ones(batch_size)
        far_amp            = np.random.uniform(0, 15.0/180.0 * np.pi, (Nk,))
        far_phase          = np.random.uniform(-np.pi,         np.pi, (Nk,))
        far                = np.multiply(far_amp, np.exp(far_phase)).astype(np.complex128).squeeze()
        trr                = np.random.uniform(3.0, 16.0, (Nk,)).astype(np.float64).squeeze()

        batch_xs   = np.zeros((batch_size,5 * Nk), dtype = np.float64)
        batch_xs_c = np.zeros((batch_size, Nk),    dtype = np.complex128)
     
        # intial seq simulation with t1t2b0 values
        #seq_data = ssad.irssfp_arrayin_data( batch_size, Nk ).set( batch_ys )    
        T1r, T2r, dfr, PDr        = ssmrf.set_par(batch_ys)
        batch_xs_c[...,0:Nk]      = ssmrf.bloch_sim_batch_cuda( batch_size, 100, Nk, PDr,\
         T1r, T2r, dfr, M0, trr, far, ti )
        #seperate real/imag parts or abs/angle parts, no noise output
        batch_xs[:,0:Nk] = np.real(batch_xs_c)
        batch_xs[:,Nk:2*Nk] = np.imag(batch_xs_c)    
        batch_xs[:,2*Nk:3*Nk] = far_amp
        batch_xs[:,3*Nk:4*Nk] = far_phase
        batch_xs[:,4*Nk:5*Nk] = trr         

        #input with noise
        #batch_xsnoise = batch_xs  + np.random.uniform(-0.01,0.01,(batch_size,2*Nk))    

        #train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
        model.train(batch_xs, batch_ys)
        if i%10 == 0:
            model.test(batch_xs, batch_ys)
        if i%1000 == 0:
            model.save('../save_data/MRF_encoder_t1t2b0')   