"""
class wrap the tensorflow model into a class
this simple neuron network has one full connection lay
this demo came from https://danijar.com/structuring-your-tensorflow-models/
The self._atrrs are wrapped 

@property
def func():
    ##do some thing###
    return self.func_a

is equivalent to

def func():
    ##do some thing###
func = property(func)  #wrap func by property()

In Python, property() is a built-in function that creates and returns a property object. 
above code is also equivalent to

# make empty property
func = property()
# assign fget
func = func.getter(func) # this func.getter get the return object of func(), which self.func_a

usage:
python classmodeldemo.py

"""
import dill
import pickle
import cPickle
import traceback
import numpy as np
import functools
import tensorflow as tf

def lazy_property(function):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    """
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            #with tf.variable_scope(function.__name):
                #setattr(self, attribute, function(self))
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

class Model:

    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.prediction
        self.optimize
        self.error

    @lazy_property
    def prediction(self):
        data_size = int(self.data.get_shape()[1])
        target_size = int(self.target.get_shape()[1])
        weight = tf.Variable(tf.truncated_normal([data_size, target_size]))
        bias = tf.Variable(tf.constant(0.1, shape=[target_size]))
        incoming = tf.matmul(self.data, weight) + bias
        return tf.nn.softmax(incoming)

    @lazy_property
    def optimize(self):
        cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction))
        optimizer = tf.train.RMSPropOptimizer(0.03)
        return optimizer.minimize(cross_entropy)


    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    def save(self,name):
        """save class as self.name.txt"""
        file = open(name+'.save','w')
        dill.dump((self.__dict__),file)
        file.close()
        return self

    def load(self,name):
        """try load self.name.txt"""
        file = open(name+'.save','r')
        self.__dict__ = dill.load(file)
        file.close()
        return self


if __name__ == '__main__':
    #mnist = input_data.read_data_sets('./MNIST_data/', one_hot=True)
    x = np.zeros((1000,784))
    y_ = np.zeros((1000,10))
    image = tf.placeholder(tf.float32, [None, 784])
    label = tf.placeholder(tf.float32, [None, 10])
    model = Model(image, label)
    model.save('modelmy')
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    for _ in range(10):
      #images, labels = mnist.test.images, mnist.test.labels
      error = sess.run(model.error, {image: x, label: y_})
      print('Test error {:6.2f}%'.format(100 * error))
      for _ in range(60):
        #images, labels = mnist.train.next_batch(100)
        sess.run(model.optimize, {image: x, label: y_})
    

