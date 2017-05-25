"""
this demo came from https://danijar.com/structuring-your-tensorflow-models/
"""

import pickle
import cPickle
import traceback
import numpy as np
import functools
import tensorflow as tf

class Model:

    def __init__(self, data, target):
        data_size = int(data.get_shape()[1])
        target_size = int(target.get_shape()[1])
        weight = tf.Variable(tf.truncated_normal([data_size, target_size]))
        bias = tf.Variable(tf.constant(0.1, shape=[target_size]))
        incoming = tf.matmul(data, weight) + bias
        self.prediction = tf.nn.softmax(incoming)
        cross_entropy = -tf.reduce_sum(target*tf.log(self.prediction))
        self.optimize = tf.train.RMSPropOptimizer(0.03).minimize(cross_entropy)
        mistakes = tf.not_equal(
            tf.argmax(target, 1), tf.argmax(self.prediction, 1))
        self.error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

    #@property
    def prediction(self):
        return self.prediction

    #@property
    def optimize(self):
        return self.optimize

    #@property
    def error(self):
        return self.error
    """
    def save(self,name):
        """save class as self.name.txt"""
        file = open(name+'.save','w')
        file.write(cPickle.dumps(self.__dict__))
        file.close()
        return self

    def load(self,name):
        """try load self.name.txt"""
        file = open(name+'.save','r')
        dataPickle = file.read()
        file.close()
        self.__dict__ = cPickle.loads(dataPickle)
        return self
    """
if __name__ == '__main__':
    #mnist = input_data.read_data_sets('./MNIST_data/', one_hot=True)
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
