"""
Full connection net example
"""
import numpy as np
import functools
import tensorflow as tf
import neural_network.tf_wrap as tf_wrap
from neural_network.tf_layer import tf_layer
import scipy.io as sio
import utilities.utilities_func as ut
import bloch_sim.sim_seq_MRF_irssfp_cuda as ssmrf
import bloch_sim.sim_utilities_func as simut



#pathdat   = '/working/larson/UTE_GRE_shuffling_recon/circus_20171208_t1t2/'
#pathdat  = '/working/larson/UTE_GRE_shuffling_recon/circus_20180112_invivo/'
#pathdat = '/working/larson/UTE_GRE_shuffling_recon/circus_20180119_invivo/pw2ms/'
pathdat = '/working/larson/UTE_GRE_shuffling_recon/circus_20180119_invivo/pw504us/'

# these functions should be defined specifically for individal neural network
# example of the prediction function, defined using tensorflow lib
def tf_prediction_func( model ):
    #if model.arg is None:
    #    model.arg = [1.0, 1.0]
    # get data size
    NNlayer     = tf_layer( w_std = 0.2 )
    data_size   = int(model.data.get_shape()[1])
    target_size = int(model.target.get_shape()[1])
    mid_size    = 256
    #y1_rn = model.data + NNlayer.full_connection(model.data, in_fc_wide = data_size, out_fc_wide = data_size,    activate_type = 'sigmoid', layer_norm = 1)
    #y2_rn = y1_rn      + NNlayer.full_connection(y1_rn,      in_fc_wide = data_size, out_fc_wide = data_size,    activate_type = 'sigmoid', layer_norm = 1)
    #y3_rn = y2_rn      + NNlayer.full_connection(y2_rn,      in_fc_wide = data_size, out_fc_wide = data_size,    activate_type = 'sigmoid', layer_norm = 1)

    # one full connection layer
    #y1 = NNlayer.full_connection(model.data, in_fc_wide = data_size, out_fc_wide = mid_size,    activate_type = 'sigmoid', layer_norm = 1)
    #y2 = NNlayer.multi_full_connection(y1, n_fc_layers = 8,            activate_type = 'sigmoid', layer_norm = 1)   
    #y  = NNlayer.full_connection_dropout(y2, arg= model.arg,        in_fc_wide = mid_size,  out_fc_wide = target_size, activate_type = 'sigmoid')

    # resnet-FC   
    y1 = NNlayer.full_connection(model.data, in_fc_wide = data_size, out_fc_wide = mid_size,    activate_type = 'ReLU', layer_norm = 1)
    y2 = y1 + NNlayer.multi_full_connection(y1, n_fc_layers = 8,            activate_type = 'ReLU', layer_norm = 1) 
    y3 = y2 + NNlayer.multi_full_connection(y2, n_fc_layers = 8,            activate_type = 'ReLU', layer_norm = 1) 
    y4 = y3 + NNlayer.multi_full_connection(y3, n_fc_layers = 8,            activate_type = 'ReLU', layer_norm = 1) 
    y  = NNlayer.full_connection_dropout(y4, arg= model.arg,        in_fc_wide = mid_size,  out_fc_wide = target_size, activate_type = 'sigmoid')
    return y

# example of the prediction function, defined using tensorflow lib
def tf_optimize_func( model ):
  # l2-norm
    #loss =  tf.reduce_sum(tf.pow(tf.subtract(model.target[:,:4],model.prediction[:,:4]),2) ) \
    #       + tf.reduce_mean(-tf.reduce_sum(model.target[:,4:] * tf.log(model.prediction[:,4:]), reduction_indices=[1]))
    loss =  tf.reduce_sum(tf.pow(tf.subtract(model.target,model.prediction),2) ) #\
          #+ tf.reduce_sum(model.prediction[:,4:] )
    return tf.train.AdamOptimizer(1e-4).minimize(loss) 

# example of the error function, defined using tensorflow lib
def tf_error_func( model ):
    model.arg =  1.0#[1.0, 1.0]
    # error as the difference between target and prediction, argmax as output layer
    mistakes = tf.reduce_sum(tf.pow(tf.subtract(model.target,model.prediction),2) )/tf.reduce_sum(tf.pow(model.target,2) )
    # error=cost(mistakes) = ||mistakes||_2
    return (tf.cast(mistakes, tf.float32))**(0.5)


#############################

def test1():
    mat_contents  = sio.loadmat(pathdat+'dict_pca_2freq.mat');#dict_pca
    coeff         = np.array(mat_contents["coeff"].astype(np.float32))

    par           = mat_contents["par"]
    
    do_coeff   = 1

    batch_size = 800
    Nk         = par[0]['irfreq'][0][0][0]#892#far.shape[0]#par.irfreq#
    if do_coeff:
        Ndiv   = coeff.shape[1]#par[0]['ndiv'][0][0][0]#16
    else:
        Ndiv   = coeff.shape[0]
    orig_Ndiv  = coeff.shape[0]//2
    npar       = 7
    model = tf_wrap.tf_model_top([None,  Ndiv], [None,  npar], tf_prediction_func, tf_optimize_func, tf_error_func, arg = 0.5)
    #model.restore(pathdat + 'test_model_save_2freq')

    #fa1        = 45.0#par[0]['fa1'][0][0][0].astype(np.float32)#35#30 #deg
    fa         = par[0]['fa'][0][0][0].astype(np.float32)
    tr         = par[0]['tr'][0][0][0].astype(np.float32)#3.932#4.337 #ms
    ti         = par[0]['ti'][0][0][0].astype(np.float32)#11.0 #ms
    te         = par[0]['te'][0][0][0].astype(np.float32)#1.5 #ms

    far, trr,ter   = simut.rftr_const(Nk, fa, tr, te)

    M0         = simut.def_M0()

    #run tensorflow on cpu, count of gpu = 0
    config     = tf.ConfigProto()#(device_count = {'GPU': 0})
    #allow tensorflow release gpu memory
    config.gpu_options.allow_growth=True
    t1t2_group = np.zeros((4, 2),dtype = np.float64)
    t1t2_group[0,0] = 600.0/5000.0#600     #0.12
    t1t2_group[0,1] = 40.0/500.0  #40     #0.08
    t1t2_group[1,0] = 1000.0/5000.0#1000   #0.2
    t1t2_group[1,1] = 80.0/500.0 #80      #0.16   
    t1t2_group[2,0] = 3000.0/5000.0 #3000  #0.6
    t1t2_group[2,1] = 200.0/500.0 #200     #0.4
    t1t2_group[3,0] = 0.0/5000.0
    t1t2_group[3,1] = 0.0/500.0 

    for i in range(1000000):
        batch_ys           = np.random.uniform(0,1,(batch_size,npar)).astype(np.float64)
        batch_ys[:,2]      = np.random.uniform(0,1.0/tr,(batch_size)).astype(np.float64)#np.zeros(batch_size)#
        batch_ys_tmp       = np.random.uniform(0,4,(batch_size))

        for k in range(batch_size):
            if batch_ys_tmp[k] <= 4 and batch_ys_tmp[k] > 3:
                batch_ys[k,0] = t1t2_group[0,0] + np.random.uniform(-0.05,0.025)                #0.07 to 0.145
                batch_ys[k,1] = t1t2_group[0,1] + np.random.uniform(-0.05,0.025)                #0.04 to 0.105
                batch_ys[k,4] = 1.0
                batch_ys[k,5] = 0.0
                batch_ys[k,6] = 0.0
            elif batch_ys_tmp[k] <= 3 and batch_ys_tmp[k] > 2:
                batch_ys[k,0] = t1t2_group[1,0] + np.random.uniform(-0.035,0.035)#-0.035,0.035  #0.145 to 0.255
                batch_ys[k,1] = t1t2_group[1,1] + np.random.uniform(-0.035,0.035)               #0.105 to 0.215
                batch_ys[k,4] = 0.0
                batch_ys[k,5] = 1.0
                batch_ys[k,6] = 0.0                            
            elif batch_ys_tmp[k] <= 2 and batch_ys_tmp[k] > 1:
                batch_ys[k,0] = t1t2_group[2,0] + np.random.uniform(-0.15,0.25)                 #0.45  to 0.85
                batch_ys[k,1] = t1t2_group[2,1] + np.random.uniform(-0.15,0.25)                 #0.55  to 0.65
                batch_ys[k,4] = 0.0
                batch_ys[k,5] = 0.0
                batch_ys[k,6] = 1.0                             
            else: 
                batch_ys[k,0] = t1t2_group[3,0] #+ np.random.uniform(-0.0,0.02)
                batch_ys[k,1] = t1t2_group[3,1] #+ np.random.uniform(-0.0,0.08) 
                batch_ys[k,4] = 0.0
                batch_ys[k,5] = 0.0
                batch_ys[k,6] = 0.0                                


        T1r, T2r, dfr, PDr = ssmrf.set_par(batch_ys[...,0:4])
        batch_xs_c1         = ssmrf.bloch_sim_batch_cuda2( batch_size, 100, Nk, PDr, T1r, T2r, dfr, M0, trr, ter, far, ti )        
        #batch_xs_c2         = ssmrf.bloch_sim_batch_cuda2( batch_size, 100, Nk, PDr, T1r, T2r, dfr, M0, trr, ter, np.absolute(far), ti )
        batch_xs_c2         = ssmrf.bloch_sim_batch_cuda2( batch_size, 100, Nk, PDr, T1r, T2r, dfr + 137.0, M0, trr, ter, far, ti )
      
        #ut.plot(np.absolute(batch_xs_c[0,:]))   
        batch_xs = np.zeros((batch_size,2*orig_Ndiv), dtype = batch_xs_c1.dtype)
        if orig_Ndiv is not Nk:
            batch_xs[:,0:orig_Ndiv]            = (simut.average_dict(batch_xs_c1, orig_Ndiv))#(np.dot(np.absolute(simut.average_dict(batch_xs_c, Ndiv)), coeff)) 
            batch_xs[:,orig_Ndiv:2*orig_Ndiv]  = (simut.average_dict(batch_xs_c2, orig_Ndiv))#(np.dot(np.absolute(simut.average_dict(batch_xs_c, Ndiv)), coeff)) 


        batch_xs = batch_xs + np.random.ranf(1)[0]*np.random.uniform(-0.002,0.002,(batch_xs.shape))
        batch_xs = np.absolute(batch_xs)

        if do_coeff:
            batch_xs = np.dot(batch_xs, coeff)

        else:
            batch_xs = batch_xs

        for dd in range(batch_xs.shape[0]):
            tc1 = batch_xs[dd,:]
    
            normtc1 = np.linalg.norm(tc1)

            if normtc1  > 0.01: 
                batch_xs[dd,:] = tc1
            else:
                batch_ys[dd,:] = np.zeros([1,npar])
        batch_xs = batch_xs/np.ndarray.max(batch_xs.flatten())

        model.test(batch_xs, batch_ys)        
        model.train(batch_xs, batch_ys)

        if i % 100 == 0:
            prey = model.prediction(batch_xs,np.zeros(batch_ys.shape))
            ut.plot(prey[...,0], batch_ys[...,0], line_type = '.', pause_close = 1)
            ut.plot(prey[...,1], batch_ys[...,1], line_type = '.', pause_close = 1)
            ut.plot(prey[...,2], batch_ys[...,2], line_type = '.', pause_close = 1)
            ut.plot(prey[...,3], batch_ys[...,3], line_type = '.', pause_close = 1)
            ut.plot(prey[...,4], batch_ys[...,4], line_type = '.', pause_close = 1)
            model.save(pathdat + 'test_model_save_2freq')

def test2():
    mat_contents     = sio.loadmat(pathdat+'im_pca_2freq.mat')#im.mat
    I                = np.array(mat_contents["I"].astype(np.float32))
    nx, ny, nz, ndiv = I.shape
    #print(I.shape)
    imall            = I.reshape([nx*ny*nz, ndiv])
    npar             = 7
    do_coeff         = 1

    if do_coeff is not 1:
        mat_contents  = sio.loadmat(pathdat+'dict_pca_2freq.mat');#dict_pca
        coeff         = np.array(mat_contents["coeff"].astype(np.float32))        
        imall         = np.absolute(np.dot(imall, np.transpose(coeff)))
        ndiv          = imall.shape[1]

    imall = imall/np.ndarray.max(imall.flatten())

    ut.plotim3(imall.reshape([nx, ny, nz, ndiv])[...,0],[5, -1],pause_close = 1)

    model   = tf_wrap.tf_model_top([None,  ndiv], [None,  npar], tf_prediction_func, tf_optimize_func, tf_error_func, arg = 1.0)
    model.restore(pathdat + 'test_model_save_2freq')

    prey    = model.prediction(imall, np.zeros([imall.shape[0],npar]))
    immatch = prey.reshape([nx, ny, nz, npar])
    #ut.plotim3(immatch[...,0],[10, -1],bar = 1, pause_close = 5)
    #ut.plotim3(immatch[...,1],[10, -1],bar = 1, pause_close = 5)
    #ut.plotim3(immatch[...,2],[10, -1],bar = 1, pause_close = 5)   
    #ut.plotim3(immatch[...,3],[10, -1],bar = 1, pause_close = 5)   
    #ut.plotim3(immatch[...,4],[10, -1],bar = 1, pause_close = 5)       
    sio.savemat(pathdat + 'MRF_cnn_matchtt_2freq.mat', {'immatch':immatch, 'imall':imall})



#if __name__ == '__main__':
    #test1()
    #test2()
