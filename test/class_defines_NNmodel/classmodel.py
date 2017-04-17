# Working example from blog post at:
# https://danijar.github.io/structuring-your-tensorflow-models
import functools
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.

    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


class Model:

    def __init__(self, image, label):
        self.image = image
        self.label = label
        self.prediction
        self.optimize
        self.error

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def prediction(self):
        x = self.image
        x = tf.contrib.slim.fully_connected(x, 200)
        x = tf.contrib.slim.fully_connected(x, 200)
        x = tf.contrib.slim.fully_connected(x, 10, tf.nn.softmax)
        return x

    @define_scope
    def optimize(self):
        logprob = tf.log(self.prediction + 1e-12)
        cross_entropy = -tf.reduce_sum(self.label * logprob)
        optimizer = tf.train.RMSPropOptimizer(0.03)
        return optimizer.minimize(cross_entropy)

    @define_scope
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.label, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))


def main():
    x = np.zeros((1000,784))
    y_ = np.zeros((1000,10))
    image = tf.placeholder(tf.float32, [None, 784])
    label = tf.placeholder(tf.float32, [None, 10])
    model = Model(image, label)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    for _ in range(10):
      #images, labels = mnist.test.images, mnist.test.labels
      error = sess.run(model.error, {image: x, label: y_})
      print('Test error {:6.2f}%'.format(100 * error))
      for _ in range(60):
        #images, labels = mnist.train.next_batch(100)
        sess.run(model.optimize, {image: x, label: y_})


if __name__ == '__main__':
  main()
