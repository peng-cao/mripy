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
import scipy
#pathdat  = '/working/larson/UTE_GRE_shuffling_recon/Mapping-MachineLearning/'
#pathdat  = '/working/larson/UTE_GRE_shuffling_recon/20170801/exp3_irssfp_largefov_invivo/'
#pathdat  = '/home/pcao/git/save_data/20170801/'
#pathdat  = '/home/pcao/git/save_data/jing_dict/'
pathdat  = '/home/pcao/git/save_data/20170814_pca_all/'
#pathdat = '/home/pcao/git/save_data/20170814/'
#pathexe  = '/home/pcao/git/mripy/test/neural_network_training/'
#pathsave = '/home/pcao/git/nn_checkpoint/'
# these functions should be defined specifically for individal neural network
# example of the prediction function, defined using tensorflow lib

def tf_prediction_func( model ):
    #if model.arg is None:
    #    model.arg = [1.0, 1.0]
    # get data size
    NNlayer     = tf_layer( w_std = 0.4 )
    data_size   = int(model.data.get_shape()[1])
    target_size = int(model.target.get_shape()[1])
    mid_size         = 256
    cnn_pool1d_len   = 2
    cnn_n_features   = 8
    n_cnn_layers     = 6
    cnn_ker_size     = 32
    cnn_out_size     = (mid_size *cnn_n_features) //(cnn_pool1d_len**n_cnn_layers)
    # one full connection layer
    # input data shape [-1, data_size],                     output data shape [-1, mid_size]
    y1      = NNlayer.full_connection(model.data, in_fc_wide = data_size, out_fc_wide = mid_size, activate_type = 'sigmoid')
    y2      = NNlayer.multi_full_connection(y1,  n_fc_layers = 1,                                 activate_type = 'sigmoid')       
    # input data shape [-1,  mid_size], output data shape [-1, mid_size, 1, 1]
    y1_4d   = tf.reshape(y2, [-1,mid_size,1,1]) #reshape into 4d tensor
    # input data shape [-1,  mid_size, 1, 1],               output data shape [-1, mid_size/2, 1, cnn_n_feature]
    #y2      = NNlayer.convolution2d(y1_4d, cov_ker_size = (5,1), in_n_features = 1, out_n_features = cnn_n_features, pool_size = [1, cnn_pool1d_len, 1, 1], activate_type = 'sigmoid')
    # input data shape [-1,  mid_size/2, 1, cnn_n_feature], output data shape [-1, mid_size/4, 1, cnn_n_feature]
    #y3      = NNlayer.convolution2d(y2,    cov_ker_size = (5,1), in_n_features = cnn_n_features, out_n_features = cnn_n_features, pool_size = [1, cnn_pool1d_len, 1, 1], activate_type = 'sigmoid')
    y3      = NNlayer.multi_convolution2d(y1_4d, cov_ker_size = (cnn_ker_size,1), n_cnn_layers = n_cnn_layers, \
                in_n_features_arr = (1,              cnn_n_features, cnn_n_features, cnn_n_features, cnn_n_features, cnn_n_features), \
               out_n_features_arr = (cnn_n_features, cnn_n_features, cnn_n_features, cnn_n_features, cnn_n_features, cnn_n_features), \
                   pool_size = [1, cnn_pool1d_len, 1, 1], activate_type = 'sigmoid')
    # input data shape [-1,  mid_size/4, 1, cnn_n_feature], output data shape [-1, cnn_out_size=cnn_n_features*mid_size//4]
    y3_flat = tf.reshape(y3, [-1, cnn_out_size]) #flatten
    y4      = NNlayer.multi_full_connection(y3_flat,  n_fc_layers = 1,                          activate_type = 'sigmoid')    
    # input data shape [-1, cnn_out_size],                  output data shape [-1, cnn_out_size]
    y       = NNlayer.full_connection(y4, in_fc_wide = cnn_out_size, out_fc_wide = target_size, activate_type = 'sigmoid')
    # input data shape [-1, cnn_out_size],                  output data shape [-1, target_size]
    #y       = NNlayer.full_connection_dropout(y4, model.arg, in_fc_wide = cnn_out_size, out_fc_wide = target_size, activate_type = 'sigmoid')
    # softmax output
    return y#tf.nn.softmax(y)

# example of the prediction function, defined using tensorflow lib
def tf_optimize_func( model ):
    #model.arg = 0.5#[0.5, 0.5]
    # cost funcion as cross entropy = y * log(y)
    #cross_entropy = tf.reduce_mean(-tf.reduce_sum(model.target * tf.log(model.prediction), reduction_indices=[1]))
    #cross_entropy = -tf.reduce_sum(model.target * tf.log(model.prediction))
    #optimizer = tf.train.RMSPropOptimizer(0.003)

     # l2-norm
    loss =  tf.reduce_sum(tf.pow(tf.subtract(model.target[:,0],model.prediction[:,0]),2) ) \
          + tf.reduce_sum(tf.pow(tf.subtract(model.target[:,1],model.prediction[:,1]),2) ) \
          + tf.reduce_sum(tf.pow(tf.subtract(model.target[:,2],model.prediction[:,2]),2) ) \
          + tf.reduce_sum(tf.pow(tf.subtract(model.target[:,3],model.prediction[:,3]),2) ) #\
          #+ 0.1* tf.reduce_sum(tf.pow(model.prediction[:,2],2)) 
    #loss =  tf.reduce_sum(tf.pow(tf.subtract(model.target[:,0],model.prediction[:,0]),2) ) \
    #      + 0.5*tf.reduce_sum(tf.pow(tf.subtract(model.target[:,1],model.prediction[:,1]),2) ) \
    #      + 0.5*tf.reduce_sum(tf.pow(tf.subtract(model.target[:,2],model.prediction[:,2]),2) ) \
    #      + tf.reduce_sum(tf.pow(tf.subtract(model.target[:,3],model.prediction[:,3]),2) ) \
    #      + 0.5 * tf.reduce_sum(tf.pow(model.prediction[:,1],2)) \
    #      + 1.5 * tf.reduce_sum(tf.pow(model.prediction[:,2],2)) 
           
    # minimization apply to cross_entropy
    return tf.train.AdamOptimizer(1e-4).minimize(loss)    #optimizer.minimize(cross_entropy) #

# example of the error function, defined using tensorflow lib
def tf_error_func( model ):
    #model.arg =  1.0#[1.0, 1.0]
    # mistakes as the difference between target and prediction, argmax as output layer
    mistakes = tf.reduce_sum(tf.pow(tf.subtract(model.target,model.prediction),2) )/tf.reduce_sum(tf.pow(model.target,2) )
    # error=cost(mistakes) = ||mistakes||_2
    return (tf.cast(mistakes, tf.float32))**(0.5)


#############################

def test1():
    mat_contents    = sio.loadmat(pathdat+'dict_pca.mat');#dict_pca
    #dictall       = np.array(mat_contents["avedictall"].astype(np.float32))
    #label         = np.array(mat_contents["dict_label"].astype(np.float32))
    
    par             = mat_contents["par"]
    
    read_coeff_flag = 1 # 0 not use coeff; 1 read coeff from mat file; 2 generate coeff by pca
    abs_flag        = 0 #apply pca on absolute time course
    batch_size      = 800
    Nk              = par[0]['irfreq'][0][0][0]#892#far.shape[0]#par.irfreq#

    if read_coeff_flag is 1:
        coeff      = np.array(mat_contents["coeff"].astype(np.float32))
        Ndiv       = coeff.shape[1]#Nk #par[0]['ndiv'][0][0][0]#16
        orig_Ndiv  = coeff.shape[0]#Nk
    elif read_coeff_flag is 2:
        Ndiv       = 20#20 pca 
        orig_Ndiv  = Nk#coeff.shape[0]#Nk        
    else:
        Ndiv       = Nk#coeff.shape[1]#Nk #par[0]['ndiv'][0][0][0]#16
        orig_Ndiv  = Nk#coeff.shape[0]#Nk

    npar       = 4
    model = tf_wrap.tf_model_top([None,  Ndiv], [None,  npar], tf_prediction_func, tf_optimize_func, tf_error_func)


    fa         = par[0]['fa'][0][0][0].astype(np.float32)#35#30 #deg
    tr         = par[0]['tr'][0][0][0].astype(np.float32)#3.932#4.337 #ms
    ti         = par[0]['ti'][0][0][0].astype(np.float32)#11.0 #ms
    #print(fa)
    #print(tr)
    #print(ti)
    #print(Nk)
    #print(Ndiv)

    far, trr   = simut.rftr_const(Nk, fa, tr)
    #far,trr    = simut.rftr_rand(Nk, fa, tr, 2*tr)
    M0         = simut.def_M0()

    #run tensorflow on cpu, count of gpu = 0
    config     = tf.ConfigProto()#(device_count = {'GPU': 0})
    #allow tensorflow release gpu memory
    config.gpu_options.allow_growth=True

    #compute pca
    if read_coeff_flag is 2:
        batch_ys           = np.random.uniform(0,1,(batch_size,4)).astype(np.float64)
        #batch_ys[:,0]      = np.random.uniform(0.1,0.6,(batch_size)).astype(np.float64)
        #batch_ys[:,1]      = np.random.uniform(0.1,0.3,(batch_size)).astype(np.float64)
        #batch_ys[:,2]      = np.random.uniform(0,1.0/tr,(batch_size)).astype(np.float64)
        #batch_ys[:,3]      = np.random.uniform(0.1,1.0,(batch_size)).astype(np.float64)# np.ones(batch_size)## np.random.uniform(0.5,1,(batch_size)).astype(np.float64)#    
        
        # intial seq simulation with t1t2b0 values
        npca               = Ndiv
        T1r, T2r, dfr, PDr = ssmrf.set_par(batch_ys)
        batch_xs_c         = ssmrf.bloch_sim_batch_cuda( batch_size, 100, Nk, PDr, T1r, T2r, dfr, M0, trr, far, ti )
        #ut.plot(np.absolute(batch_xs_c[0,:]),pause_close =1)   
        if orig_Ndiv < Nk:
            batch_xs = simut.average_dict(batch_xs_c, orig_Ndiv)#(np.dot(np.absolute(simut.average_dict(batch_xs_c, Ndiv)), coeff)) 
        else:
            batch_xs = batch_xs_c
        pca_mtx            = np.dot(np.matrix(batch_xs).getH(),batch_xs)
        U, s, V            = scipy.sparse.linalg.svds(pca_mtx,npca)
        coeff              = U[:,npca-1::-1]
        sio.savemat(pathdat + 'MRF_pca_coeff.mat', {'coeff':coeff, 'dict':batch_xs})

    for i in range(1000000):
        batch_ys           = np.random.uniform(0,1,(batch_size,4)).astype(np.float64)
        #batch_ys[:,0]      = np.random.uniform(0.1,0.6,(batch_size)).astype(np.float64)
        #batch_ys[:,1]      = np.random.uniform(0.1,0.3,(batch_size)).astype(np.float64)
        batch_ys[:,2]      = np.random.uniform(0,1.0/tr,(batch_size)).astype(np.float64)
        #batch_ys[:,3]      = np.ones(batch_size)#np.random.uniform(0.1,1.0,(batch_size)).astype(np.float64)# # np.random.uniform(0.5,1,(batch_size)).astype(np.float64)#
        #batch_ys[:,2]      = np.zeros(batch_size)
        #batch_ys[:,2]      = np.random.uniform(0.19,0.21,(batch_size)).astype(np.float64)#0.2*np.ones(batch_size)
        # intial seq simulation with t1t2b0 values
        #seq_data = ssad.irssfp_arrayin_data( batch_size, Nk ).set( batch_ys )
        T1r, T2r, dfr, PDr = ssmrf.set_par(batch_ys)
        batch_xs_c         = ssmrf.bloch_sim_batch_cuda( batch_size, 100, Nk, PDr, T1r, T2r, dfr, M0, trr, far, ti )


        #ut.plot(np.absolute(batch_xs_c[0,:]),pause_close =1)   
        if orig_Ndiv < Nk:
            batch_xs = simut.average_dict(batch_xs_c, orig_Ndiv)#(np.dot(np.absolute(simut.average_dict(batch_xs_c, Ndiv)), coeff)) 
        else:
            batch_xs = batch_xs_c
        batch_xs = batch_xs + np.random.ranf(1)[0]*np.random.uniform(-0.005,0.005,(batch_xs.shape))

        #batch_xs = batch_xs/np.ndarray.max(batch_xs.flatten())
        if read_coeff_flag is 1:
            if abs_flag:
                batch_xs = np.dot(np.absolute(batch_xs), coeff)
            else:
                batch_xs = np.absolute(np.dot(batch_xs, coeff))
        elif read_coeff_flag is 2:
            batch_xs = np.absolute(np.dot(batch_xs, coeff))          
        else:
            batch_xs = np.absolute(batch_xs)

        for dd in range(batch_xs.shape[0]):
            tc1 = batch_xs[dd,:] #- np.mean(imall[i,:])     
            normtc1 = np.linalg.norm(tc1)
            if normtc1  > 0.04 and batch_ys[dd,0]*5000 > 3*500*batch_ys[dd,1]:
                batch_xs[dd,:] = tc1#/normtc1
            else:
                batch_ys[dd,:] = np.zeros([1,npar])

        batch_xs = 1000*batch_xs
        #ut.plot(np.absolute(batch_xs[0,:]),pause_close =1)  
        #batch_ys[:,2]      = np.zeros(batch_size)
        #batch_ys[:,3]      = np.zeros(batch_size)

        model.train(batch_xs, batch_ys)
        model.test(batch_xs, batch_ys)
        if i % 100 == 0:
            prey = model.prediction(batch_xs,np.zeros(batch_ys.shape))
            ut.plot(prey[...,0], batch_ys[...,0], line_type = '.', pause_close = 1)
            ut.plot(prey[...,1], batch_ys[...,1], line_type = '.', pause_close = 1)
            ut.plot(prey[...,2], batch_ys[...,2], line_type = '.', pause_close = 1)
            ut.plot(prey[...,3], batch_ys[...,3], line_type = '.', pause_close = 1)
            model.save(pathdat + 'test_model_savecnn')

def test2():
    mat_contents     = sio.loadmat(pathdat+'im_pca.mat')#im.mat
    I                = np.array(mat_contents["I"])[:,:,25:45,:]
    if len(I.shape) == 3:
        nx, ny, ndiv = I.shape
        nz           = 1
    elif len(I.shape) == 4:
        nx, ny, nz, ndiv = I.shape

    imall            = I.reshape([nx*ny*nz, ndiv])
    npar             = 4
    abs_flag         = 0 #apply pca on absolute time course, else compute absolute value after pca
    if abs_flag is 0:
        imall = np.absolute(imall).astype(np.float32)
    else:
        imall = imall.astype(np.float32)
    #imall = imall/np.ndarray.max(imall.flatten())
    #ut.plotim3(imall.reshape(I.shape)[...,0])
    #for i in range(imall.shape[0]):
    #    tc = imall[i,:] #- np.mean(imall[i,:])        
    #    normtc = np.linalg.norm(tc)
    #    if normtc  > 1e-3:
    #        imall[i,:] = tc/normtc
    #    else:
    #        imall[i,:] = np.zeros([1,ndiv])
    imall = 1000.0*imall/np.ndarray.max(imall.flatten())#0.2

    ut.plotim3(imall.reshape(I.shape)[...,0],[10, 6],pause_close = 1)

    model   = tf_wrap.tf_model_top([None,  ndiv], [None,  npar], tf_prediction_func, tf_optimize_func, tf_error_func)
    model.restore(pathdat + 'test_model_savecnn')

    prey    = model.prediction(imall, np.zeros([imall.shape[0],npar]))
    immatch = prey.reshape([nx, ny, nz, npar])
    ut.plotim3(immatch[...,0],[10, 6],bar = 1, pause_close = 5)
    ut.plotim3(immatch[...,1],[10, 6],bar = 1, pause_close = 5)
    ut.plotim3(immatch[...,2],[10, 6],bar = 1, pause_close = 5)   
    ut.plotim3(immatch[...,3],[10, 6],bar = 1, pause_close = 5)   
      
    sio.savemat(pathdat + 'MRF_cnn_matchttt.mat', {'immatch':immatch, 'imall':imall})

"""
    mat_contents  = sio.loadmat(pathdat+'dict_pca.mat')#forNN_dict.mat
    dictall       = np.array(mat_contents["avedictall"].astype(np.float32))
    label         = np.array(mat_contents["dict_label"].astype(np.float32))
    batch_size    = 2000

    dictall = dictall/np.ndarray.max(dictall.flatten())
    for i in range(dictall.shape[0]):
        dictall[i,:] = dictall[i,:] / np.linalg.norm(dictall[i,:])

    #data   = tf.placeholder(tf.float32, [None,  13])
    #target = tf.placeholder(tf.float32, [None,  3])
    model = tf_wrap.tf_model_top([None,  13], [None,  3], tf_prediction_func, tf_optimize_func, tf_error_func)
    for i in range(5000000):
        batch_start = (i*batch_size + np.random.randint(0, 1000))%dictall.shape[0]
        batch_stop  = np.minimum(batch_start + batch_size, dictall.shape[0])

        batch_xs = dictall[batch_start:batch_stop,:]
        batch_ys = label[batch_start:batch_stop,:]
        #batch_xsnoise = batch_xs  + np.random.ranf(1)[0]*np.random.uniform(-0.05,0.05,(batch_xs.shape))

        #batch_ys[:,2] = np.random.randint(10)/10#np.random.ranf(1)[0]
        batch_xs = batch_xs * np.random.ranf(1)[0]#batch_ys[1,2] 
        #batch_xsnoise = batch_xs  + np.random.ranf(1)[0]*np.random.uniform(-0.05,0.05,(batch_xs.shape))
        #ut.plot(batch_xs[0,:].squeeze())
        model.train(batch_xs, batch_ys)#np.random.ranf(1)[0]*
        model.test(batch_xs, batch_ys)#np.random.ranf(1)[0]*
        if i % 1000 == 0:#or i == 49999
            #prey = model.prediction(batch_xs,np.zeros(batch_ys.shape))
            #ut.plot(prey[...,0], batch_ys[...,0], line_type = '.')
            #ut.plot(prey[...,1], batch_ys[...,1], line_type = '.')
            #ut.plot(prey[...,2], batch_ys[...,2], line_type = '.')
            model.save('../save_data/test_model_savett')

def test2():
    mat_contents     = sio.loadmat(pathdat+'im_pca.mat')#im.mat
    I                = np.array(mat_contents["I"].astype(np.float32))
    nx, ny, nz, ndiv = I.shape
    imall            = I.reshape([nx*ny*nz, ndiv])

    imall = imall/np.ndarray.max(imall.flatten())
    #ut.plotim3(imall.reshape(I.shape)[...,0])
    for i in range(imall.shape[0]):
        normtc = np.linalg.norm(imall[i,:])
        if normtc  > 5e-3:
            imall[i,:] = imall[i,:]/normtc
        else:
            imall[i,:] = np.zeros([1,ndiv])
    
    ut.plotim3(imall.reshape(I.shape)[...,0])

    model   = tf_wrap.tf_model_top([None,  13], [None,  3], tf_prediction_func, tf_optimize_func, tf_error_func)
    model.restore('../save_data/test_model_savett')

    prey    = model.prediction(imall, np.zeros([imall.shape[0],3]))
    immatch = prey.reshape([nx, ny, nz, 3])
    ut.plotim3(immatch[...,0])
    ut.plotim3(immatch[...,1])    
    sio.savemat(pathdat+'MRF_cnn_matchtt.mat', {'immatch':immatch, 'imall':imall})

def test3():
    mat_contents  = sio.loadmat(pathdat+'dict_pca.mat');
    dictall       = np.array(mat_contents["avedictall"].astype(np.float32))
    label         = np.array(mat_contents["dict_label"].astype(np.float32))

    dictall = dictall/np.ndarray.max(dictall.flatten())
    for i in range(dictall.shape[0]):
        dictall[i,:] = dictall[i,:] / np.linalg.norm(dictall[i,:])

    model = tf_wrap.tf_model_top([None,  13], [None,  3], tf_prediction_func, tf_optimize_func, tf_error_func)
    model.restore('../save_data/test_model_savett')
    model.test(dictall, label)
    prey = model.prediction(dictall,np.zeros(label.shape))
    ut.plot(prey[...,0], label[...,0], line_type = '.')
    ut.plot(prey[...,1], label[...,1], line_type = '.')
    ut.plot(prey[...,2], label[...,2], line_type = '.')
"""

#if __name__ == '__main__':
    #test1()
    #test2()
