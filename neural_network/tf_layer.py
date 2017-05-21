import tensorflow as tf
import numpy as np

class tf_layer:
    def __init__( self ):
    	self.config = None
    	self.arg    = None

    #intial random weight_variable
    def weight_variable( self, shape ):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

   #initial constant bias_variable
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
        return tf.nn.max_pool(x, ksize=pool_size, strides = strides, padding=padding)

    def pool( self, x, pool_size = [1, 2, 1, 1], strides = None, pool_type = 'max_pool' ):
        if strides is None:
            strides = pool_size
        if pool_type is 'max_pool':
            h_pool = self.max_pool(x)
        else:
            h_pool = x
        return h_pool

    def activate( self, x, activate_type = 'ReLU' ):
        if activate_type is 'ReLu':
            y_act = tf.nn.relu(x)
        elif activate_type is 'sigmoid':
            y_act = tf.sigmoid(x)
        elif activate_type is 'softmax':
            y_act = tf.nn.softmax(x)
        elif activate_type is 'argmax':
            y_act = tf.argmax(x)
        else:
            y_act = x
        return y_act

    def full_connection( self, x, in_fc_wide = None, out_fc_wide = None, activate_type = 'ReLU' ):
        #weighting and bias for a layer
        W_fc = self.weight_variable([in_fc_wide, out_fc_wide])
        b_fc = self.bias_variable([out_fc_wide])
        # densely connected layer
        h_fc = tf.matmul(x, W_fc) + b_fc
        y_act = self.activate( h_fc, activate_type)
        return y_act

    def full_connection_dropout( self, x, arg = 1.0, in_fc_wide = None, out_fc_wide = None, activate_type = 'ReLU' ):
        #do dropout for the input
        h_fcn_drop = tf.nn.dropout(x, arg)
        #weighting and bias for a layer
        W_fc = self.weight_variable([in_fc_wide, out_fc_wide])
        b_fc = self.bias_variable([out_fc_wide])
        # densely connected layer
        h_fc = tf.matmul(h_fcn_drop, W_fc) + b_fc
        y_act = self.activate( h_fc, activate_type)
        return y_act

    def multi_full_connection( self, x, n_fc_layer = 1, in_fc_wide_arr = None, out_fc_wide_arr = None, activate_type = 'ReLU' ):
        for i in range(n_fc_layer):
            #set fc_wide parameters
            if in_fc_wide_arr is None:
                in_fc_wide = x.shape[0]
            else:
                in_fc_wide = in_fc_wide_arr[i]

            if out_fc_wide_arr is None:
                out_fc_wide = x.shape[0]
            else:
                out_fc_wide = out_fc_wide_arr[i]
            #set input output
            if i is 0:
                yin = x
            else:
                yin = yout
            #define fc layers
            yout  = self.full_connection(yin, in_fc_wide = in_fc_wide, out_fc_wide = out_fc_wide, activate_type = 'sigmoid')    
        return yout

    def multi_full_connection_dropout( self, x, arg = 1.0, n_fc_layer = 1, in_fc_wide_arr = None, out_fc_wide_arr = None, activate_type = 'ReLU' ):
        for i in range(n_fc_layer):
            # set fc_wide parameters 
            if in_fc_wide_arr is None:
                in_fc_wide = x.shape[0]
            else:
                in_fc_wide = in_fc_wide_arr[i]

            if out_fc_wide_arr is None:
                out_fc_wide = x.shape[0]
            else:
                out_fc_wide = out_fc_wide_arr[i]
            #set input output
            if i is 0:
                yin = x
            else:
                yin = yout
            #define fc layers
            yout  = self.full_connection_dropout(yin, arg = arg, in_fc_wide = in_fc_wide, out_fc_wide = out_fc_wide, activate_type = 'sigmoid')    
        return yout

    def convolution2d( self, x, cov_ker_size = (5,1), in_n_features = 1,\
          out_n_features = 32, conv_strides = [1, 1, 1, 1],\
          pool_size = [1, 2, 1, 1], pool_type = 'max_pool', activate_type = 'ReLU' ):
        #weighting and bias
        W_conv = self.weight_variable([cov_ker_size[0], cov_ker_size[1],\
                                         in_n_features, out_n_features])
        b_conv = self.bias_variable([out_n_features])
        #define convolution 
        h_conv = self.conv2d(x, W_conv, strides = conv_strides) + b_conv
        h_pool = self.pool(h_conv, pool_size = pool_size, pool_type = pool_type) #pooling
        y_act  = self.activate( h_pool, activate_type = activate_type)
        return y_act

    
    def multi_convolution2d(self, x, n_fc_layer = 1, cov_ker_size = (5,1), in_n_features_arr = None,\
          out_n_features_arr = None, conv_strides = [1, 1, 1, 1],\
          pool_size = [1, 2, 1, 1], pool_type = 'max_pool', activate_type = 'ReLU'):
        for i in range(n_fc_layer):
            # set n_features parameters
            if in_n_features_arr is None:
                in_n_features = 1
            else:
                in_n_features = in_n_features_arr[i]

            if out_n_features_arr is None:
                out_n_features = 1
            else:
                out_n_features = out_n_features_arr[i]
            #set input output
            if i is 0:
                yin = x
            else:
                yin = yout
            # define cnn layers
            yout = self.convolution2d(yin, cov_ker_size = cov_ker_size, \
                in_n_features = in_n_features, out_n_features = out_n_features, \
                 conv_strides = conv_strides, pool_size = pool_size, pool_type = pool_type, activate_type = activate_type)
        return yout

    def deconvolution2d( self, x, cov_ker_size = (5,1), in_n_features = 32,\
                          out_n_features = 1, conv_strides = [1, 1, 1, 1],\
                          pool_size = [1, 2, 1, 1], out_shape = None,, activate_type = 'ReLU'):
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

    def multi_deconvolution2d(self, x, n_fc_layer = 1, cov_ker_size = (5,1), in_n_features_arr = None,\
          out_n_features_arr = None, conv_strides = [1, 1, 1, 1],\
          pool_size = [1, 2, 1, 1], out_shape = None, activate_type = 'ReLU'):
        for i in range(n_fc_layer):
            # set n_features parameters
            if in_n_features_arr is None:
                in_n_features = 1
            else:
                in_n_features = in_n_features_arr[i]

            if out_n_features_arr is None:
                out_n_features = 1
            else:
                out_n_features = out_n_features_arr[i]
            #set input output
            if i is 0:
                yin = x
            else:
                yin = yout
            # define cnn layers
            yout = self.deconvolution2d(yin, cov_ker_size = cov_ker_size, \
                   in_n_features = in_n_features, out_n_features = out_n_features, \
                   conv_strides  = conv_strides, pool_size = pool_size, out_shape = None,\
                   activate_type = activate_type)
        return yout
