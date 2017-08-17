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
pathdat  = '/working/larson/UTE_GRE_shuffling_recon/Mapping-MachineLearning/'
#pathexe  = '/home/pcao/git/mripy/test/neural_network_training/'
#pathsave = '/home/pcao/git/nn_checkpoint/'
# these functions should be defined specifically for individal neural network
# example of the prediction function, defined using tensorflow lib

def tf_prediction_func( model ):
    #if model.arg is None:
    #    model.arg = [1.0, 1.0]
    # get data size
    NNlayer     = tf_layer()
    data_size   = int(model.data.get_shape()[1])
    target_size = int(model.target.get_shape()[1])
    mid_size         = 128
    cnn_pool1d_len   = 2
    cnn_n_features   = 2
    cnn_out_size     = cnn_n_features*mid_size//(cnn_pool1d_len**4)
    # one full connection layer
    # input data shape [-1, data_size],                     output data shape [-1, mid_size]
    y1      = NNlayer.full_connection(model.data, in_fc_wide = data_size, out_fc_wide = mid_size, activate_type = 'tanh')
    # input data shape [-1,  mid_size], output data shape [-1, mid_size, 1, 1]
    y1_4d   = tf.reshape(y1, [-1,mid_size,1,1]) #reshape into 4d tensor
    # input data shape [-1,  mid_size, 1, 1],               output data shape [-1, mid_size/2, 1, cnn_n_feature]
    #y2      = NNlayer.convolution2d(y1_4d, cov_ker_size = (5,1), in_n_features = 1, out_n_features = cnn_n_features, pool_size = [1, cnn_pool1d_len, 1, 1], activate_type = 'sigmoid')
    # input data shape [-1,  mid_size/2, 1, cnn_n_feature], output data shape [-1, mid_size/4, 1, cnn_n_feature]
    #y3      = NNlayer.convolution2d(y2,    cov_ker_size = (5,1), in_n_features = cnn_n_features, out_n_features = cnn_n_features, pool_size = [1, cnn_pool1d_len, 1, 1], activate_type = 'sigmoid')
    y3      = NNlayer.multi_convolution2d(y1_4d, cov_ker_size = (32,1), n_cnn_layers = 4, \
                in_n_features_arr = (1,              cnn_n_features, cnn_n_features, cnn_n_features), \
               out_n_features_arr = (cnn_n_features, cnn_n_features, cnn_n_features, cnn_n_features), \
                   pool_size = [1, cnn_pool1d_len, 1, 1], activate_type = 'tanh')
    # input data shape [-1,  mid_size/4, 1, cnn_n_feature], output data shape [-1, cnn_out_size=cnn_n_features*mid_size//4]
    y3_flat = tf.reshape(y3, [-1, cnn_out_size]) #flatten
    # input data shape [-1, cnn_out_size],                  output data shape [-1, cnn_out_size]
    y      = NNlayer.full_connection(y3_flat, in_fc_wide = cnn_out_size, out_fc_wide = target_size, activate_type = 'ReLU')
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
    loss = tf.reduce_sum(tf.pow(tf.subtract(model.target,model.prediction),2) ) #+ tf.reduce_sum(tf.pow(model.prediction,2)) 
   
    # minimization apply to cross_entropy
    return tf.train.AdamOptimizer(1e-3).minimize(loss)    #optimizer.minimize(cross_entropy) #

# example of the error function, defined using tensorflow lib
def tf_error_func( model ):
    #model.arg =  1.0#[1.0, 1.0]
    # mistakes as the difference between target and prediction, argmax as output layer
    mistakes = tf.reduce_sum(tf.pow(tf.subtract(model.target,model.prediction),2) )/tf.reduce_sum(tf.pow(model.target,2) )
    # error=cost(mistakes) = ||mistakes||_2
    return (tf.cast(mistakes, tf.float32))**(0.5)


#############################

def test1():
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

#if __name__ == '__main__':
    #test1()
    #test2()
