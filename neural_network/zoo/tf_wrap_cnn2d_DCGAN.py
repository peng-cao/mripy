"""
class wrap the tensorflow model into a class tf_wrap
"""
import numpy as np
import functools
import tensorflow as tf
import neural_network.tf_wrap as tf_wrap
from neural_network.tf_layer import tf_layer
from tensorflow.examples.tutorials.mnist import input_data

# these functions should be defined specifically for individal neural network
# example of the prediction function, defined using tensorflow lib

def generator( z ):
    NNlayer     = tf_layer()
    #data_size   = int(z.get_shape()[1])
    #im_shape    = (int(z.get_shape()[1]), int(z.get_shape()[2]))
    #target_size = int(model.target.get_shape()[1])
    pool_len    = 2
    n_features  = 8
    ksize       = (5,5)
    #out_size = data_size#n_features*data_size//(pool_len**4)
    y1      = NNlayer.full_connection(z, in_fc_wide = 1, out_fc_wide = 4*4*n_features, activate_type = 'None')
    # input data shape [-1,  mid_size], output data shape [-1, mid_size, 1, 1]
    y1_4d   = tf.reshape(y1, [-1,4,4,n_features]) #reshape into 4d tensor
    #y1_4d    = tf.reshape(model.data, [-1,im_shape[0],im_shape[1],1]) #reshape into 4d tensor
    # input size   [-1, im_shape[0],          im_shape[1],          n_features ]
    # output size  [-1, im_shape[0]*pool_len, im_shape[1]*pool_len, n_features ]
    h2      = NNlayer.multi_deconvolution2d(y1_4d, cov_ker_size = ksize, n_cnn_layers = 3, \
                                           in_n_features_arr  = (n_features,   2*n_features, 4*n_features), \
                                           out_n_features_arr = (2*n_features, 4*n_features, 8*n_features), \
                                           pool_size = [1, pool_len, pool_len, 1], activate_type = 'ReLU')
    h3      = NNlayer.convolution2d(h2, cov_ker_size = ksize, in_n_features = 8*n_features, \
                                    out_n_features = 16*n_features, \
                                    pool_size = [1, pool_len, pool_len, 1], activate_type = 'tanh')
    return h3#tf.nn.tanh(h4)

def discriminator( image, reuse=False ):
    NNlayer      = tf_layer()
    pool_len     = 2
    n_features   = 8
    ksize        = (5,5)
    cnn_out_size = int(n_features * image.get_shape()[1]) * int(image.get_shape()[2])//(pool_len**4)
    if reuse:
        tf.get_variable_scope().reuse_variables()
    h1      = NNlayer.multi_convolution2d(image, cov_ker_size = ksize, n_cnn_layers = 4, \
                                           in_n_features_arr  = (16*n_features, 8*n_features, 4*n_features, 2*n_features), \
                                           out_n_features_arr = ( 8*n_features, 4*n_features, 2*n_features,   n_features), \
                                           pool_size = [1, pool_len, pool_len, 1], activate_type = 'ReLU')
    y2      = tf.reshape(h1, [-1, cnn_out_size]) #flatten
    y3      = NNlayer.full_connection(y2, in_fc_wide = cnn_out_size, out_fc_wide = 1, activate_type = 'None')
    return tf.nn.sigmoid(y3), y3

def tf_prediction_func( model ):
    NNlayer                 = tf_layer()
    outy.G                  = generator(model.data)
    outy.D,  outy.D_logits  = discriminator(model.target)
    outy.D_, outy.D_logits_ = discriminator(outy.G, reuse=True)
    # softmax output
    return outy#tf.nn.softmax(y)

# example of the prediction function, defined using tensorflow lib
def tf_optimize_func( model ):
    gety = model.prediction
    d_loss_real = tf.reduce_mean(\
        tf.nn.sigmoid_cross_entropy_with_logits(gety.D_logits,\
                                                tf.ones_like(gety.D)))
    d_loss_fake = tf.reduce_mean(\
        tf.nn.sigmoid_cross_entropy_with_logits(gety.D_logits_,\
                                                tf.zeros_like(gety.D_)))
    d_loss = d_loss_real + d_loss_fake

    g_loss = tf.reduce_mean(\
        tf.nn.sigmoid_cross_entropy_with_logits(gety.D_logits_,\
                                                tf.ones_like(gety.D_)))
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'd_' in var.name]
    g_vars = [var for var in t_vars if 'g_' in var.name]

    if model.arg == 'train_D':
    d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                    .minimize(d_loss, var_list=d_vars)
    elif model.arg == 'train_G':
    g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                    .minimize(g_loss, var_list=g_vars)
    return #optimizer.minimize(loss)

# example of the error function, defined using tensorflow lib
def tf_error_func( model ):
    #model.arg = 1.0#[1.0, 1.0]
    #training accuracy
    gety = model.prediction
    g_loss = tf.reduce_mean(\
        tf.nn.sigmoid_cross_entropy_with_logits(gety.D_logits_,\
                                                tf.ones_like(gety.D_)))
    return g_loss

#############################

def test1():
    mnist  = input_data.read_data_sets('./data/MNIST_data/', one_hot=True)
    data   = tf.placeholder(tf.float32, [None, 28,28,1])
    target = tf.placeholder(tf.float32, [None])
    model  = tf_wrap.tf_model_top(data, target, tf_prediction_func, tf_optimize_func, tf_error_func)
    for _ in range(100):
        model.test(mnist.test.images, mnist.test.images)
        for _ in range(100):
            batch_xs, batch_ys = mnist.train.next_batch(1000)
            model.set_arg('train_D').train(batch_image, batch_z)
            model.set_arg('train_G').train(batch_image, batch_z)
    model.save('../save_data/test_model_save')

def test2():
    mnist  = input_data.read_data_sets('./data/MNIST_data/', one_hot=True)
    data   = tf.placeholder(tf.float32, [None, 28,28,1])
    target = tf.placeholder(tf.float32, [None])
    model  = tf_wrap.tf_model_top(data, target, tf_prediction_func, tf_optimize_func, tf_error_func)
    model.restore('../save_data/test_model_save')
    model.test(mnist.test.images, mnist.test.images)
#if __name__ == '__main__':
    #test1()
    #test2()
