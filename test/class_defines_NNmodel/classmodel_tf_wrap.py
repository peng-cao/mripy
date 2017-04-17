"""
class wrap the tensorflow model into a class
this simple neuron network, inspired by the demo code from https://danijar.com/structuring-your-tensorflow-models/
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
python test.py

"""
import dill
import pickle
import cPickle
import traceback
import numpy as np
import functools
import tensorflow as tf

# define the model in abstract form
class model_abstract:
    def __init__( self, prediction_func, optimize_func, error_func ):
        """define the neural network in abstract form, need prediction, optimize, error functions as inputs"""
        self._prediction_func = prediction_func
        self._optimize_func   = optimize_func
        self._error_func      = error_func

    def prediction( self, data, target ):
        return self._prediction_func(data, target)

    def optimize( self, data, target ):
        return self._optimize_func(target, self.prediction(data, target))

    def error( self, data, target ):
        return self._error_func(target, self.prediction(data, target))

    def save( self, name ):
        """save class as self.name.txt"""
        file = open(name+'.save','w')
        dill.dump((self.__dict__),file)
        file.close()
        return self

    def load( self, name ):
        """try load self.name.txt"""
        file = open(name+'.save','r')
        self.__dict__ = dill.load(file)
        file.close()
        return self

# define model wrap for tensorflow lib
def lazy_property( function ):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    """
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator( self ):
        if not hasattr(self, attribute):
            #with tf.variable_scope(function.__name):
                #setattr(self, attribute, function(self))
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator

#wrapped model using tensorflow lib
#model contain data, target and model_abstract
class tf_model_wrap:
    def __init__( self, data, target, model ):
        self.data       = data
        self.target     = target
        self.mode       = model
        self.prediction
        self.optimize
        self.error

    @lazy_property
    def prediction( self ):
        return self.model.prediction(self.data, self.target)

    @lazy_property
    def optimize( self ):
        return self.model.optimize(self.data, self.target)

    @lazy_property
    def error( self ):
        return self.model.error(self.data, self.target)

#define the top level model that contains training and testing functions using tensorflow lib
class tf_model_top:
    # intialize tensorflow model
    def __init__( self, data_shape, target_shape, tf_prediction_func, tf_optimize_func, tf_error_func ):
        # model first defined in abstract form, which contains prediction, optimize, error functions
        self.model      = model_abstract(tf_prediction_func, tf_optimize_func, tf_error_func)
        # tensorflow style data and target defination, as inputs to model
        self.data       = tf.placeholder(tf.float32, data_shape) # e.g. [None, 784]
        self.target     = tf.placeholder(tf.float32, target_shape) # e.g. [None, 10]
        # put data, target and model together
        self.model_wrap = tf_model_wrap(self.data, self.target, self.model)
        self.sess       = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    # train neural network, using all training data, do mini-batch in this function
    def train_all_batch( self, train_data, train_target, N_example, N_batch, mini_batch_func ):
        #mini-batch
        batch_x, batch_y = mini_batch_func(N_example, N_batch, train_data, train_target)
        for _ in range(N_example//N_batch):
            self.sess.run(self.model_wrap.optimize, {self.data: batch_x, self.target: batch_y})
        return self

    # simple training function, do one step training, should be putted in a loop for mini-batch
    def train( self, train_data, train_target ):
        self.sess.run(self.model_wrap.optimize, {self.data: train_data, self.target: train_target})
        return self

    # simple training function, do one step training, should be putted in a loop for mini-batch
    def prediction( self, data ):
        target = self.sess.run(self.model_wrap.prediction, {self.data: data})
        return target

    # test neural network using testing data
    def test( self, test_data, test_target ):
        error = self.sess.run(self.model_wrap.error, {self.data: test_data, self.target: test_target})
        print('Test error {:6.2f}%'.format(100 * error))
        return self

    # save the tensorflow model
    def save( self, name ):
        saver = tf.train.Saver(tf.all_variables())
        saver.save(self.sess, name)
        return self

    # restore the tensorflow model
    def restore( self, name ):
        nsaver = tf.train.Saver(tf.all_variables())
        nsaver.restore(self.sess, name)
        return self

# these functions should be defined specifically for individal neural network
# example of the prediction function, defined using tensorflow lib
def tf_prediction_func( data, target ):
    # get data size
    data_size   = int(data.get_shape()[1])
    target_size = int(target.get_shape()[1])
    # one full connection layer
    weight      = tf.Variable(tf.truncated_normal([data_size, target_size]))
    bias        = tf.Variable(tf.constant(0.1, shape=[target_size]))
    # y = data * W + b
    y    = tf.matmul(data, weight) + bias
    # softmax output
    return tf.nn.softmax(y)

# example of the prediction function, defined using tensorflow lib
def tf_optimize_func( target, prediction ):
    # cost funcion as cross entropy = y * log(y)
    cross_entropy = -tf.reduce_sum(target * tf.log(prediction))
    # optimizer using RMSPropOptimizer
    optimizer     = tf.train.RMSPropOptimizer(0.03)
    # minimization apply to cross_entropy
    return optimizer.minimize(cross_entropy)

# example of the error function, defined using tensorflow lib
def tf_error_func( target, prediction ):
    # mistakes as the difference between target and prediction, argmax as output layer
    mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
    # error=cost(mistakes) = ||mistakes||_2
    return tf.reduce_mean(tf.cast(mistakes, tf.float32))

def test1():
    #mnist = input_data.read_data_sets('./MNIST_data/', one_hot=True)
    x = np.zeros((1000,784))
    y_ = np.zeros((1000,10))
    model = tf_model_top([None, 784], [None, 10], tf_prediction_func, tf_optimize_func, tf_error_func)
    for _ in range(10):
        #model.test(images, labels)
      for _ in range(60):
        images, labels = mnist.train.next_batch(100)
        model.train(images, labels)

#if __name__ == '__main__':
    #test1()
