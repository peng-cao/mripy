"""
DCGAN example, not tested yet
inspired by the code at http://bamos.github.io/2016/08/09/deep-completion/
"""
import numpy as np
import functools
import tensorflow as tf
import neural_network.tf_wrap as tf_wrap
from neural_network.tf_layer import tf_layer
from tensorflow.examples.tutorials.mnist import input_data

# from z to fake image
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
    return h3

# from image to label or logits
def discriminator( image, reuse = False ):
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
    # z should be sampled from a noise prior, for creating fake image
    G             = generator(model.data['z'])
    # D  = sigmoid(D_logits),  D_logits: D(x),    x is image/model.data
    # D_ = sigmoid(D_logits_), D_logits: D(G(z)), z is sampled from noise prior
    D,  D_logits  = discriminator(model.data['image'])
    D_, D_logits_ = discriminator(outy.G, reuse = True)
    # struncture output
    outy = {'G':G,'D':D, 'D_':D_,'D_logits':D_logits,'D_logits_':D_logits_}
    return outy#tf.nn.softmax(y)

# example of the prediction function, defined using tensorflow lib
def tf_optimize_func( model ):
    #run prediction
    gety = model.prediction
    D    = gety['D']
    D_   = gety['D_']
    D_logits  = gety['D_logits']
    D_logits_ = gety['D_logits_']
    #define loss for discriminator
    # discriminator maximizing probablity for D(x)   with label = 1, i.e. image labeled as 1
    #                                    and D(G(z)) with label = 0, i.e. fake image labeled as 0
    # d_loss_real: cross_entropy = label * -log(sigmoid(D(x))), lables = ones
    #                            = -log(sigmoid(D(x))), in D(x), x image
    # d_loss_real = tf.reduce_mean(-tf.log(D))
    # d_loss_fake: cross_entropy = (1-labels) * -log(1-sigmoid(D(G(z)))), labels = zeros
    #                            = -log(1-sigmoid(D(G(z)))), in D(G(z)),
    #                              z sampled from a noise prior, G(z) fake image
    # d_loss_fake = tf.reduce_mean(-tf.log(1 - D_))
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logits,\
                                                tf.ones_like(D)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logits_,\
                                                tf.zeros_like(D_)))
    d_loss = d_loss_real + d_loss_fake
    #define loss for generator,
    # generator maximizing probability for D(G(z)) with label = 1 version,
    # i.e. maximizing the probability of fake image been labeled as 1 by discriminator
    #g_loss:  cross_entropy = label * -log(sigmoid(D(x))), lables = ones
    #                       = -log(sigmoid(D(G(z)))), G(z), fake image
    # g_loss = tf.reduce_mean(-tf.log(D_))
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logits_,\
                                                tf.ones_like(D_)))
    # generator minimizing probability for D(G(z)) with label = 0 version,
    # i.e. minimizing the probability for fake image been labeled as 0 by discriminator
    #g_loss:  -1 * cross_entropy = -1 * (1-labels) * -log(1-sigmoid(D(G(z)))), labels = zeros
    #                            = log(1-sigmoid(D(G(z)))), G(z), fake image
    # g_loss = tf.reduce_mean(tf.log(1 - D_))
    #g_loss = -tf.reduce_mean(\
    #    tf.nn.sigmoid_cross_entropy_with_logits(D_logits_,\
    #                                            tf.zeros_like(D_)))

    #select variables for g_ and d_, i.e. training generator and discriminator seperately
    #t_vars = tf.trainable_variables()
    #d_vars = [var for var in t_vars if 'd_' in var.name]
    #g_vars = [var for var in t_vars if 'g_' in var.name]
    # selectively do training for generator and discriminator
    if model.arg == 'train_D':
        optim = tf.train.AdamOptimizer(1e-4).minimize(d_loss) #, var_list = d_vars
    elif model.arg == 'train_G':
        optim = tf.train.AdamOptimizer(1e-4).minimize(g_loss) #, var_list = g_vars
    return optim#optimizer.minimize(loss)

# example of the error function, defined using tensorflow lib
def tf_error_func( model ):
    #model.arg = 1.0#[1.0, 1.0]
    #training accuracy
    gety = model.prediction
    D_logits_ = gety['D_logits_']
    D_        = gety['D_']
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logits_,\
                                                tf.ones_like(D_)))
    return g_loss

#############################

def test1():
    mnist  = input_data.read_data_sets('./data/MNIST_data/', one_hot=True)
    image  = tf.placeholder(tf.float32, [None, 28,28,1])
    z      = tf.placeholder(tf.float32, [None])
    model  = tf_wrap.tf_model_top({'image':image,'z':z}, None, tf_prediction_func, tf_optimize_func, tf_error_func)
    for _ in range(100):
        model.test(mnist.test.images, mnist.test.images)
        for _ in range(100):
            batch_size = 1000
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_z = np.random.uniform(0,1,(batch_size,)).astype(np.float32)
            model.set_arg('train_D').train({'image':batch_image,'z':batch_z}, None)
            model.set_arg('train_G').train({'image':batch_image,'z':batch_z}, None)
    model.save('../save_data/test_model_save')

def test2():
    mnist   = input_data.read_data_sets('./data/MNIST_data/', one_hot=True)
    image  = tf.placeholder(tf.float32, [None, 28,28,1])
    z      = tf.placeholder(tf.float32, [None])
    model   = tf_wrap.tf_model_top({'image':image,'z':z}, None, tf_prediction_func, tf_optimize_func, tf_error_func)
    batch_z = np.random.uniform(0,1,(mnist.test.images.shape[0],)).astype(np.float32)
    model.restore('../save_data/test_model_save')
    model.test({'image':mnist.test.images, 'z':batch_z}, None)
#if __name__ == '__main__':
    #test1()
    #test2()
