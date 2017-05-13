import tensorflow as tf
import numpy as np

class tf_layer:
    def __init__( self ):
    	self.config = None
    	self.arg    = None

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)    

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)    

    #define convolution and max pooling
    def conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME' ):
        return tf.nn.conv2d(x, W, strides=strides, padding=padding)    

    def max_pool(x, ksize=[1, 2, 1, 1], strides=None, padding='SAME'):
        if strides is None:
            strides = ksize
        return tf.nn.max_pool(x, ksize=ksize, strides = strides, padding=padding)

    def pool( self, x, ksize = [1, 2, 1, 1], strides = None, pool_type = 'max_pool' ):
        if strides is None:
            strides = ksize

        if pool_type is 'max_pool':
            h_pool = self.max_pool(x)  
        else:
            h_pool = x
        return h_pool

    def activate( self, x, activate_type = 'ReLU' ):
        if pool_type is 'ReLu':
            y_act = self.nn.relu(x)

        elif pool_type is 'sigmoid':    
            y_act = self.sigmoid(x)

        else:
            y_act = x

        return y_act

    def full_conection ( self, x, activate = 'ReLU' ):
        return y_out

    def convolution2d ( self, x, wide_cov_ker = (1,0) , n_features = 32, pool_type = 'max_pool', activate_type = 'ReLU' ):
        #weighting and bias in the first convolution,
        W_conv = self.weight_variable([wide_cov_ker[0], wide_cov_ker[1], 1, n_features]) #32 features
        b_conv = self.bias_variable([n_features])      
        h_conv = conv2d(x_image, W_conv) + b_conv
        h_pool = self.pool( h_conv, pool_type = pool_type ) #pooling
        y_act  = self.activate( h_pool, activate_type = activate_type)
        return y_act

    def trans_convolution ( self, x, pool = 'max_pool', activate = 'ReLU' ):
        return y_out



