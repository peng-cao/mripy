import tensorflow as tf
import numpy as np

class tf_layer:
    def __init__( self ):
    	self.config = None
    	self.arg    = None

    #intial weight_variable
    def weight_variable( self, shape ):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

   #initial bias_variable
    def bias_variable( self, shape ):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    #convolution
    def conv2d( self, x, W, strides=[1, 1, 1, 1], padding='SAME' ):
        return tf.nn.conv2d(x, W, strides=strides, padding=padding)
    #deconvolution, or transpose of convolution
    def deconv2d( self, x, W, out_shape, strides=[1, 1, 1, 1], padding='SAME' ):
        return tf.nn.conv2d_transpose(x, W, out_shape, strides=strides, padding=padding)

    def max_pool( self, x, pool_size=[1, 2, 1, 1], strides=None, padding='SAME' ):
        if strides is None:
            strides = pool_size
        return tf.nn.max_pool(x, ksize=ksize, strides = strides, padding=padding)

    def pool( self, x, pool_size = [1, 2, 1, 1], strides = None, pool_type = 'max_pool' ):
        if strides is None:
            strides = pool_size
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
        elif pool_type is 'argmax':
            y_act = tf.argmax(x)
        else:
            y_act = x
        return y_act

    def full_connection( self, x, fc_wide = None, activate_type = 'ReLU' ):
        #weighting and bias for a layer
        W_fc = weight_variable([x.shape[-1], fc_wide])
        b_fc = bias_variable([fc_wide])
        # densely connected layer
        h_fc = tf.matmul(x, W_fc) + b_fc
        y_act = self.activate( h_fc, activate_type)
        return y_act

    def full_connection_dropout( self, x, arg, fc_wide = None, activate_type = 'ReLU' ):
        #do dropout for the input
        h_fcn_drop = tf.nn.dropout(x, arg)
        #weighting and bias for a layer
        W_fc = weight_variable([x.shape[-1], fc_wide])
        b_fc = bias_variable([fc_wide])
        # densely connected layer
        h_fc = tf.matmul(h_fcn_drop, W_fc) + b_fc
        y_act = self.activate( h_fc, activate_type)
        return y_act

    def convolution2d( self, x, cov_ker_size = (5,1) , in_n_features = 1,\
          out_n_features = 32, conv_strides = [1, 1, 1, 1],\
           pool_type = 'max_pool', activate_type = 'ReLU' ):
        #weighting and bias
        W_conv = self.weight_variable([cov_ker_size[0], cov_ker_size[1],\
                                         in_n_features, out_n_features])
        b_conv = self.bias_variable([out_n_features])
        h_conv = self.conv2d(x_image, W_conv, strides = conv_strides) + b_conv
        h_pool = self.pool( h_conv, pool_type = pool_type ) #pooling
        y_act  = self.activate( h_pool, activate_type = activate_type)
        return y_act

    def deconvolution2d( self, x, cov_ker_size = (5,1), in_n_features = 1,\
                          out_n_features = 32, conv_strides = [1, 1, 1, 1],\
                          pool_size = [1, 2, 1, 1], out_shape = None, activate = 'ReLU' ):
        if out_shape is None:
            out_shape     = x.shape
            out_shape[0]  = pool_size[0] * out_shape[0]
            out_shape[1]  = pool_size[1] * out_shape[1]
            out_shape[-1] = out_n_features
        #weighting and bias
        W_deconv = self.weight_variable([cov_ker_size[0], cov_ker_size[1],\
                                         in_n_features, out_n_features])
        b_deconv = self.bias_variable([out_n_features])
        h_deconv = self.deconv2d(x, W_deconv, out_shape = out_shape, \
                               strides = conv_strides) + b_deconv
        y_act  = self.activate( h_deconv, activate_type = activate_type)
        return y_act
