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
def tf_prediction_func( model ):
    #if model.arg is None:
    #    model.arg = [1.0, 1.0]
    # get data size
    NNlayer     = tf_layer()
    data_size   = int(model.data.get_shape()[1])
    target_size = int(model.target.get_shape()[1])
    mid_size         = 256
    cnn_pool1d_len   = 2
    cnn_n_features   = 2
    cnn_out_size     = cnn_n_features*mid_size//(cnn_pool1d_len**2)
    # one full connection layer
    # input data shape [-1, data_size],                     output data shape [-1, mid_size] 
    y1      = NNlayer.full_connection(model.data, in_fc_wide = data_size, out_fc_wide = mid_size, activate_type = 'sigmoid')
    # input data shape [-1,  mid_size], output data shape [-1, mid_size, 1, 1]
    y1_4d   = tf.reshape(y1, [-1,mid_size,1,1]) #reshape into 4d tensor
    # input data shape [-1,  mid_size, 1, 1],               output data shape [-1, mid_size/2, 1, cnn_n_feature]
    #y2      = NNlayer.convolution2d(y1_4d, cov_ker_size = (5,1), in_n_features = 1, out_n_features = cnn_n_features, pool_size = [1, cnn_pool1d_len, 1, 1], activate_type = 'sigmoid')
    # input data shape [-1,  mid_size/2, 1, cnn_n_feature], output data shape [-1, mid_size/4, 1, cnn_n_feature]
    #y3      = NNlayer.convolution2d(y2,    cov_ker_size = (5,1), in_n_features = cnn_n_features, out_n_features = cnn_n_features, pool_size = [1, cnn_pool1d_len, 1, 1], activate_type = 'sigmoid')
    y3      = NNlayer.multi_convolution2d(y1_4d, cov_ker_size = (5,1), n_cnn_layers = 2, \
                in_n_features_arr = (1,              cnn_n_features), \
               out_n_features_arr = (cnn_n_features, cnn_n_features), \
                   pool_size = [1, cnn_pool1d_len, 1, 1], activate_type = 'sigmoid')
    # input data shape [-1,  mid_size/4, 1, cnn_n_feature], output data shape [-1, cnn_out_size=cnn_n_features*mid_size//4]
    y3_flat = tf.reshape(y3, [-1, cnn_out_size]) #flatten
    # input data shape [-1, cnn_out_size],                  output data shape [-1, cnn_out_size]
    y4      = NNlayer.full_connection(y3_flat, in_fc_wide = cnn_out_size, out_fc_wide = cnn_out_size, activate_type = 'sigmoid')
    # input data shape [-1, cnn_out_size],                  output data shape [-1, target_size]
    y       = NNlayer.full_connection_dropout(y4, model.arg, in_fc_wide = cnn_out_size, out_fc_wide = target_size, activate_type = 'sigmoid')
    # softmax output
    return tf.nn.softmax(y)

# example of the prediction function, defined using tensorflow lib
def tf_optimize_func( model ):
    model.arg = 0.5#[0.5, 0.5]
    # cost funcion as cross entropy = y * log(y)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(model.target * tf.log(model.prediction), reduction_indices=[1]))
    #cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction))
    optimizer = tf.train.RMSPropOptimizer(0.03)
    # minimization apply to cross_entropy
    return optimizer.minimize(cross_entropy)

# example of the error function, defined using tensorflow lib
def tf_error_func( model ):
    model.arg =  1.0#[1.0, 1.0]
    # mistakes as the difference between target and prediction, argmax as output layer
    mistakes = tf.not_equal(tf.argmax(model.target, 1), tf.argmax(model.prediction, 1))
    # error=cost(mistakes) = ||mistakes||_2
    return tf.reduce_mean(tf.cast(mistakes, tf.float32))


#############################

def test1():
    mnist = input_data.read_data_sets('./data/MNIST_data/', one_hot=True)
    model = tf_wrap.tf_model_top([None, 784], [None, 10], tf_prediction_func, tf_optimize_func, tf_error_func, arg = 0.5)
    for _ in range(100):
        model.test(mnist.test.images, mnist.test.labels)
        for _ in range(100):
            batch_xs, batch_ys = mnist.train.next_batch(1000)            
            model.train(batch_xs, batch_ys)
    model.save('../save_data/test_model_save')

def test2():
    mnist = input_data.read_data_sets('./data/MNIST_data/', one_hot=True)
    model = tf_wrap.tf_model_top([None, 784], [None, 10], tf_prediction_func, tf_optimize_func, tf_error_func, arg = 1.0)
    model.restore('../save_data/test_model_save')
    model.test(mnist.test.images, mnist.test.labels)
#if __name__ == '__main__':
    #test1()
    #test2()
