"""
Full connection net example
"""
import numpy as np
import functools
import tensorflow as tf
import neural_network.tf_wrap as tf_wrap
from neural_network.tf_layer import tf_layer
from tensorflow.examples.tutorials.mnist import input_data
import scipy.io as sio
import utilities.utilities_func as ut
import bloch_sim.sim_seq_MRF_irssfp_cuda as ssmrf
import bloch_sim.sim_utilities_func as simut

#pathdat  = '/working/larson/UTE_GRE_shuffling_recon/Mapping-MachineLearning'
#pathdat  = '/working/larson/UTE_GRE_shuffling_recon/20170801/exp3_irssfp_largefov_invivo/'
#pathdat  = '/home/pcao/git/save_data/20170801/'
#pathdat  = '/home/pcao/git/save_data/jing_dict/'
#pathdat  = '/home/pcao/git/save_data/20170814/'
#pathdat  = '/home/pcao/git/save_data/20170814_2/'
#pathdat  = '/home/pcao/git/save_data/20170814_4/'
#pathdat  = '/home/pcao/git/save_data/20170814_5/'
#pathdat  = '/home/pcao/git/save_data/20170801_5/'
pathdat  = '/home/pcao/git/save_data/jing_dict_5/'
#pathdat   = '/home/pcao/git/save_data/20170921_6/'
#pathsave = '/home/pcao/git/nn_checkpoint/'
# these functions should be defined specifically for individal neural network
# example of the prediction function, defined using tensorflow lib
def tf_prediction_func( model ):
    #if model.arg is None:
    #    model.arg = [1.0, 1.0]
    # get data size
    NNlayer     = tf_layer( w_std = 0.2 )
    data_size   = int(model.data.get_shape()[1])
    target_size = int(model.target.get_shape()[1])
    mid_size    = 1024
    #y1_rn = model.data + NNlayer.full_connection(model.data, in_fc_wide = data_size, out_fc_wide = data_size,    activate_type = 'sigmoid', layer_norm = 1)
    #y2_rn = y1_rn      + NNlayer.full_connection(y1_rn,      in_fc_wide = data_size, out_fc_wide = data_size,    activate_type = 'sigmoid', layer_norm = 1)
    #y3_rn = y2_rn      + NNlayer.full_connection(y2_rn,      in_fc_wide = data_size, out_fc_wide = data_size,    activate_type = 'sigmoid', layer_norm = 1)

    # one full connection layer
    y1 = NNlayer.full_connection(model.data, in_fc_wide = data_size, out_fc_wide = mid_size,    activate_type = 'sigmoid', layer_norm = 1)
    y2 = NNlayer.multi_full_connection(y1, n_fc_layers = 8,            activate_type = 'sigmoid', layer_norm = 1)   
    y  = NNlayer.full_connection_dropout(y2, arg= model.arg,        in_fc_wide = mid_size,  out_fc_wide = target_size, activate_type = 'sigmoid')
    #y   = NNlayer.multi_full_connection(model.data, n_fc_layers = 12, \
    #                                    in_fc_wide_arr  = (data_size, mid_size, mid_size, mid_size, mid_size, mid_size, mid_size, mid_size, mid_size, mid_size, mid_size, mid_size), \
    #                                   out_fc_wide_arr = (mid_size,  mid_size, mid_size, mid_size, mid_size, mid_size, mid_size, mid_size, mid_size, mid_size, mid_size, target_size), \
    #                                    activate_type = 'sigmoid')
    #y   = NNlayer.multi_full_connection(model.data, n_fc_layers = 8, \
    #                                    in_fc_wide_arr = (data_size,    data_size//2, data_size//4, data_size//8,  data_size//16, data_size//32, data_size//64, data_size//128),\
    #                                   out_fc_wide_arr = (data_size//2, data_size//4, data_size//8, data_size//16, data_size//32, data_size//64, data_size//128, target_size),\
    #                                     activate_type = 'ReLU')
    # softmax output
    # softmax output
    return y#tf.nn.softmax(y)

# example of the prediction function, defined using tensorflow lib
def tf_optimize_func( model ):
    #model.arg = [0.5, 0.5]
    # cost funcion as cross entropy = y * log(y)
    #cross_entropy = tf.reduce_mean(-tf.reduce_sum(model.target * tf.log(model.prediction), reduction_indices=[1]))
    #cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction))
    #optimizer = tf.train.RMSPropOptimizer(0.003)

     # l2-norm
  # l2-norm
    loss =  tf.reduce_sum(tf.pow(tf.subtract(model.target[:,0],model.prediction[:,0]),2) ) \
          + tf.reduce_sum(tf.pow(tf.subtract(model.target[:,1],model.prediction[:,1]),2) ) \
          + tf.reduce_sum(tf.pow(tf.subtract(model.target[:,2],model.prediction[:,2]),2) ) \
          + tf.reduce_sum(tf.pow(tf.subtract(model.target[:,3],model.prediction[:,3]),2) ) #\
          #+ tf.reduce_sum(tf.pow(model.prediction[:,:],2)) #\
          #- tf.reduce_sum(tf.pow(model.prediction[:,0],2)) 
    #loss =  tf.reduce_sum(tf.pow(tf.subtract(model.target[:,0],model.prediction[:,0]),2) ) \
    #      + 0.5*tf.reduce_sum(tf.pow(tf.subtract(model.target[:,1],model.prediction[:,1]),2) ) \
    #      + 0.5*tf.reduce_sum(tf.pow(tf.subtract(model.target[:,2],model.prediction[:,2]),2) ) \
    #      + tf.reduce_sum(tf.pow(tf.subtract(model.target[:,3],model.prediction[:,3]),2) ) \
    #      + 0.5 * tf.reduce_sum(tf.pow(model.prediction[:,1],2)) \
    #      + 1.5 * tf.reduce_sum(tf.pow(model.prediction[:,2],2)) 
        
    # minimization apply to cross_entropy
    return tf.train.AdamOptimizer(1e-4).minimize(loss) #tf.train.RMSPropOptimizer(1e-4).minimize(loss)#   #optimizer.minimize(cross_entropy)

# example of the error function, defined using tensorflow lib
def tf_error_func( model ):
    model.arg =  1.0#[1.0, 1.0]
    # mistakes as the difference between target and prediction, argmax as output layer
    mistakes = tf.reduce_sum(tf.pow(tf.subtract(model.target,model.prediction),2) )/tf.reduce_sum(tf.pow(model.target,2) )
    # error=cost(mistakes) = ||mistakes||_2
    return (tf.cast(mistakes, tf.float32))**(0.5)


#############################

def test1():
    mat_contents  = sio.loadmat(pathdat+'dict_pca.mat');#dict_pca
    #dictall       = np.array(mat_contents["avedictall"].astype(np.float32))
    #label         = np.array(mat_contents["dict_label"].astype(np.float32))
    coeff         = np.array(mat_contents["coeff"].astype(np.float32))
    #cn_orders     = np.array(mat_contents["cn_orders"].astype(np.float32))
    par           = mat_contents["par"]

    batch_size = 800
    Nk         = par[0]['irfreq'][0][0][0]#892#far.shape[0]#par.irfreq#
    Ndiv       = coeff.shape[1]#par[0]['ndiv'][0][0][0]#16
    orig_Ndiv  = coeff.shape[0] 
    npar       = 4
    model = tf_wrap.tf_model_top([None,  Ndiv], [None,  npar], tf_prediction_func, tf_optimize_func, tf_error_func, arg = 0.5)


    fa         = par[0]['fa'][0][0][0].astype(np.float32)#35#30 #deg
    tr         = par[0]['tr'][0][0][0].astype(np.float32)#3.932#4.337 #ms
    ti         = par[0]['ti'][0][0][0].astype(np.float32)#11.0 #ms
    #print(fa)
    #print(tr)
    #print(ti)
    #print(Nk)
    #print(Ndiv)

    far, trr   = simut.rftr_const(Nk, fa, tr)
    M0         = simut.def_M0()

    #run tensorflow on cpu, count of gpu = 0
    config     = tf.ConfigProto()#(device_count = {'GPU': 0})
    #allow tensorflow release gpu memory
    config.gpu_options.allow_growth=True


    for i in range(1000000):
        batch_ys           = np.random.uniform(0,1,(batch_size,4)).astype(np.float64)
        #batch_ys[:,0]      = batch_ys[:,0] + 1.0*batch_ys[:,1]/10.0
        #batch_ys[:,0]      = np.random.uniform(0.07,1.0,(batch_size)).astype(np.float64)
        #batch_ys[:,1]      = np.random.uniform(0.0,0.2,(batch_size)).astype(np.float64)
        batch_ys[:,2]      = np.random.uniform(0,1.0/tr,(batch_size)).astype(np.float64)
        #batch_ys[:,2]      = np.zeros(batch_size)
        #batch_ys[:,3]      = np.ones(batch_size)#np.random.uniform(0.4,1,(batch_size)).astype(np.float64)#

        #batch_ys[:,0] = np.round(batch_ys[:,0]*20)/20
        #batch_ys[:,1] = np.round(batch_ys[:,1]*20)/20
        #batch_ys[:,2] = np.round(batch_ys[:,2]*20)/20
        #batch_ys[:,3] = np.round(batch_ys[:,3]*5)/5
        #batch_ys[:,3] = np.round(batch_ys[:,3]*5)/5
        # intial seq simulation with t1t2b0 values
        #seq_data = ssad.irssfp_arrayin_data( batch_size, Nk ).set( batch_ys )
        T1r, T2r, dfr, PDr = ssmrf.set_par(batch_ys)
        batch_xs_c         = ssmrf.bloch_sim_batch_cuda( batch_size, 100, Nk, PDr, T1r, T2r, dfr, M0, trr, far, ti )


        #ut.plot(np.absolute(batch_xs_c[0,:]))   
        if orig_Ndiv is not Nk:
            batch_xs = np.absolute(simut.average_dict(batch_xs_c, orig_Ndiv))#(np.dot(np.absolute(simut.average_dict(batch_xs_c, Ndiv)), coeff)) 
            #batch_xs  = np.absolute(simut.average_dict_cnorders(batch_xs_c, cn_orders)) #np.absolute(np.dot(batch_xs_c, cn_orders))
        else:
            batch_xs = np.absolute(batch_xs_c)


        #ut.plot(np.absolute(batch_xs[0,:]))  
        batch_xt = batch_xs
        batch_xs = batch_xs + np.random.ranf(1)[0]*np.random.uniform(-0.1,0.1,(batch_xs.shape))

        #batch_xs = batch_xs/np.ndarray.max(batch_xs.flatten())
        if 1:
            batch_xs = np.dot(batch_xs, coeff)
            batch_xt = np.dot(batch_xt, coeff)
        else:
            batch_xs = batch_xs
            batch_xt = batch_xs
        #batch_ys[:,3]      = np.zeros(batch_size)
        for dd in range(batch_xs.shape[0]):
            tc1 = batch_xs[dd,:] #- np.mean(imall[i,:])
            tc2 = batch_xt[dd,:]        
            normtc1 = np.linalg.norm(tc1)
            normtc2 = np.linalg.norm(tc2)
            if normtc2  > 0.1: #and batch_ys[dd,0]*5000 > 3*500*batch_ys[dd,1] and batch_ys[dd,0]*5000 < 20*500*batch_ys[dd,1]
                batch_xs[dd,:] = tc1#/normtc1
                batch_xt[dd,:] = tc2#/normtc2
            else:
        #        batch_xs[dd,:] = np.zeros([1,Ndiv])
        #        batch_xt[dd,:] = np.zeros([1,Ndiv])
                batch_ys[dd,:] = np.zeros([1,npar])

        batch_xs = batch_xs/np.ndarray.max(batch_xs.flatten())
        batch_xt = batch_xt/np.ndarray.max(batch_xt.flatten())
        #for kk in range(batch_xs.shape[1]):
        #    batch_xs [:,kk] = (batch_xs[:,kk])/np.std(batch_xs[:,kk] )#- np.mean(batch_xs[:,kk])
        #    batch_xt [:,kk] = (batch_xt[:,kk])/np.std(batch_xt[:,kk] )#- np.mean(batch_xt[:,kk])

        #ut.plot(np.real(batch_xs[0,:]),pause_close = 1)

        #batch_ys[:,3]      = np.ones(batch_size) * np.random.ranf(1)[0]  
        #batch_xs = batch_xs *  batch_ys[0,3] #* np.random.ranf(1)[0]#
        model.test(batch_xs, batch_ys)        
        model.train(batch_xt, batch_ys)
        model.train(batch_xs, batch_ys)

        if i % 100 == 0:
            prey = model.prediction(batch_xs,np.zeros(batch_ys.shape))
            ut.plot(prey[...,0], batch_ys[...,0], line_type = '.', pause_close = 1)
            ut.plot(prey[...,1], batch_ys[...,1], line_type = '.', pause_close = 1)
            ut.plot(prey[...,2], batch_ys[...,2], line_type = '.', pause_close = 1)
            ut.plot(prey[...,3], batch_ys[...,3], line_type = '.', pause_close = 1)
            model.save(pathdat + 'test_model_save')

def test2():
    mat_contents     = sio.loadmat(pathdat+'im_pca.mat')#im.mat
    I                = np.array(mat_contents["I"].astype(np.float32))
    nx, ny, nz, ndiv = I.shape
    #print(I.shape)
    imall            = I.reshape([nx*ny*nz, ndiv])
    npar             = 4
    imall = 0.3*imall/np.ndarray.max(imall.flatten())
    #ut.plotim3(imall.reshape(I.shape)[...,0])
    #for i in range(imall.shape[0]):
    #    tc = imall[i,:] #- np.mean(imall[i,:])        
    #    normtc = np.linalg.norm(tc)
    #    if normtc  > 1e-3:
    #        imall[i,:] = tc/normtc
    #    else:
    #        imall[i,:] = np.zeros([1,ndiv])
    #imall =imall/np.ndarray.max(imall.flatten())#0.2

    #for kk in range(imall.shape[1]):
    #    imall [:,kk] = (imall[:,kk])/np.std(imall[:,kk])# - np.mean(imall[:,kk])

    ut.plotim3(imall.reshape(I.shape)[...,0],[5, -1],pause_close = 1)

    model   = tf_wrap.tf_model_top([None,  ndiv], [None,  npar], tf_prediction_func, tf_optimize_func, tf_error_func, arg = 1.0)
    model.restore(pathdat + 'test_model_save')

    prey    = model.prediction(imall, np.zeros([imall.shape[0],npar]))
    immatch = prey.reshape([nx, ny, nz, npar])
    ut.plotim3(immatch[...,0],[10, -1],bar = 1, pause_close = 5)
    ut.plotim3(immatch[...,1],[10, -1],bar = 1, pause_close = 5)
    ut.plotim3(immatch[...,2],[10, -1],bar = 1, pause_close = 5)   
    ut.plotim3(immatch[...,3],[10, -1],bar = 1, pause_close = 5)   
      
    sio.savemat(pathdat + 'MRF_cnn_matchtt.mat', {'immatch':immatch, 'imall':imall})

"""
def test3():
    mat_contents  = sio.loadmat(pathdat+'dict_pca.mat');
    dictall       = np.array(mat_contents["avedictall"].astype(np.float32))
    label         = np.array(mat_contents["dict_label"].astype(np.float32))

    #dictall = dictall/np.ndarray.max(dictall.flatten())
    #for i in range(dictall.shape[0]):
    #    tc = dictall[i,:] - np.mean(dictall[i,:])
    #    dictall[i,:] = tc / np.linalg.norm(tc)
    dictall = 1000*dictall#/np.ndarray.max(dictall.flatten())


    model = tf_wrap.tf_model_top([None,  13], [None,  3], tf_prediction_func, tf_optimize_func, tf_error_func)
    model.restore(pathdat + 'test_model_save')
    model.test(dictall, label)
    prey = model.prediction(dictall,np.zeros(label.shape))
    ut.plot(prey[...,0], label[...,0], line_type = '.')
    ut.plot(prey[...,1], label[...,1], line_type = '.')
    ut.plot(prey[...,2], label[...,2], line_type = '.')
"""
#if __name__ == '__main__':
    #test1()
    #test2()
