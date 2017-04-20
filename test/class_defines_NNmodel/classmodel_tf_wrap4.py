"""
class wrap the tensorflow model into a class tf_wrap
"""
import numpy as np
import functools
import tensorflow as tf
import neural_network.tf_wrap as tf_wrap
from tensorflow.examples.tutorials.mnist import input_data

# these functions should be defined specifically for individal neural network
# example of the prediction function, defined using tensorflow lib
def tf_prediction_func( model ):
    #if model.arg is None:
    #    model.arg = [1.0, 1.0]
    # get data size
    data_size   = int(model.data.get_shape()[1])
    target_size = int(model.target.get_shape()[1])
    # one full connection layer
    weight      = tf.Variable(tf.truncated_normal([data_size, target_size]))
    bias        = tf.Variable(tf.constant(0.1, shape=[target_size]))
    # y = data * W + b
    y    = tf.nn.sigmoid(tf.matmul(model.data, weight) + bias)
    # softmax output
    return tf.nn.softmax(y)

# example of the prediction function, defined using tensorflow lib
def tf_optimize_func( model ):
    #model.arg = [0.5, 0.5]
    # cost funcion as cross entropy = y * log(y)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(model.target * tf.log(model.prediction), reduction_indices=[1]))
    #cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction))
    optimizer = tf.train.RMSPropOptimizer(0.03)
    # minimization apply to cross_entropy
    return optimizer.minimize(cross_entropy)

# example of the error function, defined using tensorflow lib
def tf_error_func( model ):
    #model.arg = [1.0, 1.0]
    # mistakes as the difference between target and prediction, argmax as output layer
    mistakes = tf.not_equal(tf.argmax(model.target, 1), tf.argmax(model.prediction, 1))
    # error=cost(mistakes) = ||mistakes||_2
    return tf.reduce_mean(tf.cast(mistakes, tf.float32))


#############################

def test1():
    mnist = input_data.read_data_sets('./MNIST_data/', one_hot=True)
    model = tf_wrap.tf_model_top([None, 784], [None, 10], tf_prediction_func, tf_optimize_func, tf_error_func)
    for _ in range(100):
        model.test(mnist.test.images, mnist.test.labels)
        for _ in range(100):
            batch_xs, batch_ys = mnist.train.next_batch(1000)            
            model.train(batch_xs, batch_ys)
    model.save('test_model_save')

def test2():
    mnist = input_data.read_data_sets('./MNIST_data/', one_hot=True)
    model = tf_wrap.tf_model_top([None, 784], [None, 10], tf_prediction_func, tf_optimize_func, tf_error_func)
    model.restore('test_model_save')
    model.test(mnist.test.images, mnist.test.labels)
#if __name__ == '__main__':
    #test1()
    #test2()
