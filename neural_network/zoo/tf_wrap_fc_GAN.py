"""
DCGAN example, working version
inspired by the code at http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/
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
def linear(input, output_dim, scope=None, stddev=1.0):
    norm  = tf.random_normal_initializer(stddev=stddev)
    const = tf.constant_initializer(0.0)
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable('w', [input.get_shape()[1], output_dim], initializer=norm)
        b = tf.get_variable('b', [output_dim], initializer=const)
        return tf.matmul(input, w) + b


def generator(input, h_dim):
    h0 = tf.nn.softplus(linear(input, h_dim, 'g0'))
    h1 = linear(h0, 1, 'g1')
    return h1


def discriminator(input, h_dim):
    h0 = tf.tanh(linear(input, h_dim * 2, 'd0'))
    h1 = tf.tanh(linear(h0, h_dim * 2, 'd1'))
    h2 = tf.tanh(linear(h1, h_dim * 2, scope='d2'))

    h3 = tf.sigmoid(linear(h2, 1, scope='d3'))
    return h3

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
    # z should be sampled from a noise prior, for creating fake image
    with tf.variable_scope('Gen'):
        model.z  = tf.placeholder(tf.float32, shape=(model.arg, 1))
        G        = generator(model.target, h_dim)
    with tf.variable_scope('Disc') as scope:
        model.x  = tf.placeholder(tf.float32, shape=(model.arg, 1))
        D1       = discriminator(model.data, h_dim)
        scope.reuse_variables()
        D2       = discriminator(G, h_dim)#
    return G, D1, D2#tf.nn.softmax(y)

# example of the prediction function, defined using tensorflow lib
def tf_optimize_func( model ):
    #run prediction
    G,D1,D2 = model.prediction
    model.z = model.data
    model.x = model.target

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


class DATA:
    def __init__( self, x, z, batch_size, G, D1, D2 ):
        self.x    = x
        self.z    = z
        self.data = DataDistribution()
        self.gen  = GeneratorDistribution(range=8)
        self.batch_size = batch_size
        self.G    = G
        self.D1   = D1
        self.D2   = D2

    def _samples(self, session, num_points=10000, num_bins=100):
        '''
        Return a tuple (db, pd, pg), where db is the current decision
        boundary, pd is a histogram of samples from the data distribution,
        and pg is a histogram of generated samples.
        '''
        xs = np.linspace(-self.gen.range, self.gen.range, num_points)
        bins = np.linspace(-self.gen.range, self.gen.range, num_bins)

        # decision boundary
        db = np.zeros((num_points, 1))
        for i in range(num_points // self.batch_size):
            db[self.batch_size * i:self.batch_size * (i + 1)] = session.run(self.D1, {
                self.x: np.reshape(
                    xs[self.batch_size * i:self.batch_size * (i + 1)],
                    (self.batch_size, 1)
                )
            })

        # data distribution
        d = self.data.sample(num_points)
        pd, _ = np.histogram(d, bins=bins, density=True)

        # generated samples
        zs = np.linspace(-self.gen.range, self.gen.range, num_points)
        g = np.zeros((num_points, 1))
        for i in range(num_points // self.batch_size):
            g[self.batch_size * i:self.batch_size * (i + 1)] = session.run(self.G, {
                self.z: np.reshape(
                    zs[self.batch_size * i:self.batch_size * (i + 1)],
                    (self.batch_size, 1)
                )
            })
        pg, _ = np.histogram(g, bins=bins, density=True)

        return db, pd, pg

    def _plot_distributions(self, session):
        db, pd, pg = self._samples(session)
        db_x = np.linspace(-self.gen.range, self.gen.range, len(db))
        p_x = np.linspace(-self.gen.range, self.gen.range, len(pd))
        f, ax = plt.subplots(1)
        ax.plot(db_x, db, label='decision boundary')
        ax.set_ylim(0, 1)
        plt.plot(p_x, pd, label='real data')
        plt.plot(p_x, pg, label='generated data')
        plt.title('1D Generative Adversarial Network')
        plt.xlabel('Data values')
        plt.ylabel('Probability density')
        plt.legend()
        plt.show()

#############################

def test1():
    batch_size = 12  
    model      = tf_wrap.tf_model_top((batch_size, 1), (batch_size, 1), tf_prediction_func, tf_optimize_func, tf_error_func, arg = batch_size, dtype = np.float32)

    d_loss,g_loss, dopt, gopt = model.model_wrap.optimize
    G, D1, D2                 = model.model_wrap.prediction
    ganinst                   = DATA(model.data, model.target, batch_size, G, D1, D2)
    for i in range(1200):
        x         = ganinst.data.sample(batch_size).reshape((batch_size,1)).astype(np.float32)
        z         = ganinst.gen.sample(batch_size).reshape((batch_size,1)).astype(np.float32)
        
        loss_d, _ = model.sess.run([d_loss, dopt], {model.data:x, model.target:z})
        z         = ganinst.gen.sample(batch_size).reshape((batch_size,1)).astype(np.float32)      
        loss_g, _ = model.sess.run([g_loss, gopt], {model.data:x, model.target:z})
        print('{}: {}\t{}'.format(i, loss_d, loss_g))
    ganinst._plot_distributions(model.sess)
    model.save('../save_data/test_model_save')
#if __name__ == '__main__':
    #test1()
    #test2()
