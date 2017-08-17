"""
DCGAN example, working version
based on the code at http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/
and at http://bamos.github.io/2016/08/09/deep-completion/
"""
import numpy as np
import functools
import tensorflow as tf
import neural_network.tf_wrap as tf_wrap
from neural_network.tf_layer import tf_layer
from tensorflow.examples.tutorials.mnist import input_data
import utilities.utilities_func as ut
import matplotlib.pyplot as plt


seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)


class DataDistribution(object):
    def __init__(self):
        self.mu = 4
        self.sigma = 0.5

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        samples.sort()
        return samples


class GeneratorDistribution(object):
    def __init__(self, range):
        self.range = range

    def sample(self, N):
        return np.linspace(-self.range, self.range, N) + \
            np.random.random(N) * 0.01


"""
NNlayer      = tf_layer()
# from z to fake image
def generator( z, h_dim ):
    #out_size = data_size#n_features*data_size//(pool_len**4)
    h0      = NNlayer.full_connection(z, out_fc_wide = h_dim, activate_type = 'softplus', scope = 'g0')
    # input data shape [-1,  mid_size], output data shape [-1, mid_size, 1, 1]
    h1      = NNlayer.full_connection(h0, out_fc_wide = 1, activate_type = 'None', scope = 'g1')
    return h1#tf.reshape(y5, [-1, im_shape[0], im_shape[1], 1])

# from image to label or logits
def discriminator( data, h_dim, reuse = False ):
    #NNlayer = tf_layer()
    h0      = NNlayer.full_connection(data, out_fc_wide = 2*h_dim, scope='d0', activate_type = 'tanh', w_stddev = 1.0)
    h1      = NNlayer.full_connection(h0,   out_fc_wide = 2*h_dim, scope='d1', activate_type = 'tanh', w_stddev = 1.0)
    h2      = NNlayer.full_connection(h1,   out_fc_wide = 2*h_dim,scope= 'd2', activate_type = 'tanh', w_stddev = 1.0)
    h3      = NNlayer.full_connection(h2,   out_fc_wide = 2*h_dim,scope= 'd3', activate_type = 'tanh', w_stddev = 1.0)
    #h4      = NNlayer.full_connection(h3,   out_fc_wide = 2*h_dim,scope= 'd4', activate_type = 'tanh', w_stddev = 1.0)
    #h5      = NNlayer.full_connection(h4,   out_fc_wide = 2*h_dim,scope= 'd5', activate_type = 'tanh', w_stddev = 1.0)
    #h6      = NNlayer.full_connection(h5,   out_fc_wide = 2*h_dim,scope= 'd6', activate_type = 'tanh', w_stddev = 1.0)
    h7      = NNlayer.full_connection(h3,   out_fc_wide = 1, scope='d7', activate_type = 'sigmoid', w_stddev = 1.0) 
    return h7
"""
def linear(input, output_dim, scope=None, stddev=0.1):
    norm  = tf.random_normal_initializer(stddev=stddev)
    const = tf.constant_initializer(0.0)
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable('w', [input.get_shape().as_list()[1], output_dim], initializer=norm)
        b = tf.get_variable('b', [output_dim], initializer=const)
        return tf.matmul(input, w) + b


def generator(input, h_dim):
    h0 = tf.nn.softplus(linear(input, h_dim, 'g0'))
    h1 = linear(h0, 28*28, 'g1')
    return h1


def discriminator(input, h_dim):
    h0 = tf.tanh(linear(input, h_dim * 2, 'd0'))
    h1 = tf.tanh(linear(h0, h_dim * 2, 'd1'))
    h2 = tf.tanh(linear(h1, h_dim * 2, scope='d2'))

    h3 = tf.sigmoid(linear(h2, 1, scope='d3'))
    return h3

"""
# from z to fake image
def generator( z, h_dim ):
    NNlayer     = tf_layer()
    #data_size   = int(z.get_shape()[1])
    #im_shape    = (int(z.get_shape()[1]), int(z.get_shape()[2]))
    #target_size = int(model.target.get_shape()[1])
    pool_len    = 2
    n_features  = 32
    ksize       = (5,5)
    #out_size = data_size#n_features*data_size//(pool_len**4)
    y1      = NNlayer.full_connection(z, in_fc_wide = 1, out_fc_wide = 7*7*n_features, activate_type = 'ReLU')
    # input data shape [-1,  mid_size], output data shape [-1, mid_size, 1, 1]
    y1_4d   = tf.reshape(y1, [-1,7,7,n_features]) #reshape into 4d tensor
    #y1_4d    = tf.reshape(model.data, [-1,im_shape[0],im_shape[1],1]) #reshape into 4d tensor
    # input size   [-1, im_shape[0],          im_shape[1],          n_features ]
    # output size  [-1, im_shape[0]*pool_len, im_shape[1]*pool_len, n_features ]
    h2      = NNlayer.multi_deconvolution2d(y1_4d, cov_ker_size = ksize, n_cnn_layers = 1, \
                                           in_n_features_arr  = (n_features,), \
                                           out_n_features_arr = (2*n_features,), \
                                           conv_strides = [1, pool_len, pool_len, 1], activate_type = 'ReLU')
    h3      = NNlayer.deconvolution2d(h2, cov_ker_size = ksize, in_n_features = 2*n_features, \
                                    out_n_features = 1, \
                                    conv_strides = [1, pool_len, pool_len, 1], activate_type = 'tanh')
    y   = tf.reshape(h3, [-1,28*28])
    return h3

# from x to label or logits
def discriminator( x, h_dim ):
    NNlayer      = tf_layer()
    pool_len     = 2
    n_features   = 32
    ksize        = (5,5)
    cnn_out_size = 2*n_features*(28*28)//(pool_len**4)
    y1_4d   = tf.reshape(x, [-1,28,28,1])
    h1      = NNlayer.multi_convolution2d(y1_4d, cov_ker_size = ksize, n_cnn_layers = 2, \
                                           in_n_features_arr  = (1,            n_features), \
                                           out_n_features_arr = ( n_features, 2*n_features), \
                                           pool_size = [1, pool_len, pool_len, 1], activate_type = 'ReLU')
    y2      = tf.reshape(h1, [-1, cnn_out_size]) #flatten
    y3      = NNlayer.full_connection(y2, in_fc_wide = cnn_out_size, out_fc_wide = 1, activate_type = 'sigmoid')
    return y3

def optimizer( loss, var_list, initial_learning_rate ):
    decay           = 0.95
    num_decay_steps = 150
    batch           = tf.Variable(0)
    learning_rate   = tf.train.exponential_decay(
        initial_learning_rate,
        batch,
        num_decay_steps,
        decay,
        staircase  = True
    )
    optimizer      = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss,
        global_step=batch,
        var_list   =var_list
    )
    return optimizer 

def tf_prediction_func( model ):
    h_dim        = 10
    # z should be sampled from a noise prior, for creating fake x
    with tf.variable_scope('Gen'):
        #model.z  = tf.placeholder(tf.float32, shape=(model.arg, 1))
        G        = generator(model.target, h_dim)
    with tf.variable_scope('Disc') as scope:
        #model.x  = tf.placeholder(tf.float32, shape=(model.arg, 28*28))
        D1       = discriminator(model.data, h_dim)
        scope.reuse_variables()
        D2       = discriminator(G, h_dim)#
    return G, D1, D2#tf.nn.softmax(y)
"""

# example of the prediction function, defined using tensorflow lib
def tf_optimize_func( model ):
    #run prediction
    G,D1,D2 = model.prediction
    #model.z = model.data
    #model.x = model.target

    learning_rate = 0.03
    d_loss  = tf.reduce_mean(-tf.log(D1)-tf.log(1 - D2))
    g_loss  = tf.reduce_mean(-tf.log(D2))
    #select variables for g_ and d_, i.e. training generator and discriminator seperately
    d_vars  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Disc')
    g_vars  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Gen')    

    # selectively do training for generator and discriminator
    dopt = optimizer(d_loss, d_vars, learning_rate)
    gopt = optimizer(g_loss, g_vars, learning_rate) 
    return d_loss,g_loss, dopt, gopt#optimizer.minimize(loss)

# example of the error function, defined using tensorflow lib
def tf_error_func( model ):
    #model.arg = 1.0#[1.0, 1.0]
    #training accuracy
    G,D1,D2 = model.prediction
    d_loss = tf.reduce_mean(-tf.log(D1)-tf.log(1 - D2))
    g_loss = tf.reduce_mean(-tf.log(D2))
    return d_loss,g_loss

#############################

def test1():
    mnist      = input_data.read_data_sets('./data/MNIST_data/', one_hot=True)
    batch_size = 200  
    model      = tf_wrap.tf_model_top([None,  784], [None, 1],\
                                       tf_prediction_func, tf_optimize_func, tf_error_func, \
                                       arg = batch_size, dtype = np.float32)
    d_loss,g_loss, dopt, gopt = model.model_wrap.optimize
    G, D1, D2                 = model.model_wrap.prediction
    for i in range(1200):
        x, y      = mnist.train.next_batch(batch_size)
        z         = np.random.uniform(0,1,(batch_size,1)).astype(np.float32)
        
        loss_d, _ = model.sess.run([d_loss, dopt], {model.data:x, model.target:z})
        z         = np.random.uniform(0,1,(batch_size,1)).astype(np.float32)
        loss_g, _ = model.sess.run([g_loss, gopt], {model.data:x, model.target:z})
        print('{}: {}\t{}'.format(i, loss_d, loss_g))
        if i%1000 == 0:
            getG      = model.sess.run(G, {model.data:x, model.target:z}).reshape((batch_size,28,28))
            ut.plotim3(sum(getG,0))
    model.save('../save_data/test_model_save')

#def test2():
#    mnist   = input_data.read_data_sets('./data/MNIST_data/', one_hot=True)
#    model   = tf_wrap.tf_model_top([None, 28,28,1], [None,1], tf_prediction_func, tf_optimize_func, tf_error_func)
#    batch_z = np.random.uniform(0,1,(mnist.test.images.shape[0],)).astype(np.float32)
#    model.restore('../save_data/test_model_save')
#    model.test({'image':mnist.test.images, 'z':batch_z}, None)
#if __name__ == '__main__':
    #test1()
    #test2()
