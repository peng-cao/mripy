import tensorflow as tf
import numpy as np
from math import ceil
class tf_layer:
    def __init__( self, debug = 0, w_std = 0.1, b_const = 0.1 ):
    	self.config  = None
    	self.arg     = None
        self.debug   = debug
        self.w_std   = w_std
        self.b_const = b_const

    #intial random weight_variable
    def weight_variable( self, shape ):
        initial = tf.truncated_normal(shape, stddev=self.w_std)
        return tf.Variable(initial)

   #initial constant bias_variable
    def bias_variable( self, shape ):
        initial = tf.constant(self.b_const, shape=shape)
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

    def pool( self, x, pool_size = [1, 1, 1, 1], strides = None, pool_type = 'max_pool' ):
        if strides is None:
            strides = pool_size
        if pool_type is 'max_pool':
            h_pool = self.max_pool(x, pool_size, strides)
        elif pool_type is 'ave_pool':
            h_pool = self.ave_pool(x, pool_size, strides)

        else:
            h_pool = x
        if self.debug is 1 and pool_type is not 'None':
            h_pool = tf.Print(h_pool, [tf.shape(h_pool)],
                               message='Shape of output in pooling ',
                               summarize=4, first_n=1)
        return h_pool

    def activate( self, x, activate_type = 'ReLU' ):
        if activate_type is 'ReLU':
            y_act = tf.nn.relu(x)
        elif activate_type is 'sigmoid':
            y_act = tf.sigmoid(x)
        elif activate_type is 'softmax':
            y_act = tf.nn.softmax(x)
        elif activate_type is 'argmax':
            y_act = tf.argmax(x)
        elif activate_type is 'tanh':
            y_act = tf.tanh(x)
        else:
            y_act = x
        return y_act

    def full_connection( self, x, in_fc_wide = None, out_fc_wide = None, activate_type = 'ReLU', layer_norm = None ):
        if self.debug is 1:
            x = tf.Print(x, [tf.shape(x)],
                               message='Shape of input in fc',
                               summarize=4, first_n=1)
        #weighting and bias for a layer
        W_fc = self.weight_variable([in_fc_wide, out_fc_wide])
        b_fc = self.bias_variable([out_fc_wide])
        # densely connected layer
        h_fc = tf.matmul(x, W_fc) + b_fc
        if layer_norm is not None:
            h_ln = tf.contrib.layers.layer_norm(h_fc, center=True, scale=True)
        else:
            h_ln = h_fc
        y_act = self.activate( h_ln, activate_type)
        if self.debug is 1:
            y_act = tf.Print(y_act, [tf.shape(y_act)],
                               message='Shape of output in fc',
                               summarize=4, first_n=1)
        return y_act

    def full_connection_dropout( self, x, arg = 1.0, in_fc_wide = None, out_fc_wide = None, activate_type = 'ReLU', layer_norm = None ):
        #do dropout for the input
        h_fcn_drop = tf.nn.dropout(x, arg)
        #weighting and bias for a layer
        W_fc = self.weight_variable([in_fc_wide, out_fc_wide])
        b_fc = self.bias_variable([out_fc_wide])
        # densely connected layer
        h_fc = tf.matmul(h_fcn_drop, W_fc) + b_fc
        if layer_norm is not None:
            h_ln = tf.contrib.layers.layer_norm(h_fc, center=True, scale=True)
        else:
            h_ln = h_fc
        y_act = self.activate( h_ln, activate_type)
        return y_act

    def multi_full_connection( self, x, n_fc_layers = 1, in_fc_wide_arr = None, out_fc_wide_arr = None, activate_type = 'ReLU', layer_norm = None ):
        for i in range(n_fc_layers):
            #set fc_wide parameters
            if in_fc_wide_arr is None:
                in_fc_wide = x.get_shape().as_list()[1]#x.shape[0]
            else:
                in_fc_wide = in_fc_wide_arr[i]

            if out_fc_wide_arr is None:
                out_fc_wide = x.get_shape().as_list()[1]#x.shape[0]
            else:
                out_fc_wide = out_fc_wide_arr[i]
            #set input output
            if i is 0:
                yin = x
            else:
                yin = yout
            #define fc layers
            yout  = self.full_connection(yin, in_fc_wide = in_fc_wide, out_fc_wide = out_fc_wide, activate_type =activate_type, layer_norm = layer_norm )
        return yout

    def multi_full_connection_dropout( self, x, arg = 1.0, n_fc_layers = 1, in_fc_wide_arr = None, out_fc_wide_arr = None, activate_type = 'ReLU', layer_norm = None):
        for i in range(n_fc_layers):
            # set fc_wide parameters
            if in_fc_wide_arr is None:
                in_fc_wide = x.get_shape().as_list()[1]#x.shape[0]
            else:
                in_fc_wide = in_fc_wide_arr[i]

            if out_fc_wide_arr is None:
                out_fc_wide = x.get_shape().as_list()[1]#x.shape[0]
            else:
                out_fc_wide = out_fc_wide_arr[i]
            #set input output
            if i is 0:
                yin = x
            else:
                yin = yout
            #define fc layers
            yout  = self.full_connection_dropout(yin, arg = arg, in_fc_wide = in_fc_wide, out_fc_wide = out_fc_wide, activate_type = 'ReLU', layer_norm = layer_norm)
        return yout
    
    def multi_full_connection_residual( self, x, n_fc_layers = 1, in_fc_wide_arr = None, out_fc_wide_arr = None, activate_type = 'ReLU', layer_norm = None ):
        for i in range(n_fc_layers):
            #set fc_wide parameters
            if in_fc_wide_arr is None:
                in_fc_wide = x.get_shape().as_list()[1]#x.shape[0]
            else:
                in_fc_wide = in_fc_wide_arr[i]

            if out_fc_wide_arr is None:
                out_fc_wide = x.get_shape().as_list()[1]#x.shape[0]
            else:
                out_fc_wide = out_fc_wide_arr[i]
            #set input output
            if i is 0:
                yin = x
            else:
                yin = yout
            #define fc layers
            yout  = yin + self.full_connection(yin, in_fc_wide = in_fc_wide, out_fc_wide = out_fc_wide, activate_type =activate_type, layer_norm = layer_norm )
        return yout

    def convolution2d( self, x, cov_ker_size = (5,1), in_n_features = 1,\
                       out_n_features = 1, conv_strides = [1, 1, 1, 1],\
                       pool_size = [1, 1, 1, 1], pool_type = 'max_pool', activate_type = 'ReLU', layer_norm = None ):
        if self.debug is 1:
            x = tf.Print(x, [tf.shape(x)],
                               message='Shape of input in conv ',
                               summarize=4, first_n=1)
        #weighting and bias
        W_conv = self.weight_variable([cov_ker_size[0], cov_ker_size[1],\
                                         in_n_features, out_n_features])
        b_conv = self.bias_variable([out_n_features])
        #define convolution
        h_conv = self.conv2d(x, W_conv, strides = conv_strides) + b_conv
        h_pool = self.pool(h_conv, pool_size = pool_size, pool_type = pool_type) #pooling

        if layer_norm is not None:
            h_ln = tf.contrib.layers.layer_norm(h_pool, center=True, scale=True)
        else:
            h_ln = h_pool
        y_act  = self.activate( h_ln, activate_type = activate_type)
        if self.debug is 1:
            y_act = tf.Print(y_act, [tf.shape(y_act)],
                               message='Shape of output in conv ',
                               summarize=4, first_n=1)
        return y_act


    def multi_convolution2d( self, x, n_cnn_layers = 1, cov_ker_size = (5,1), in_n_features_arr = None,\
                             out_n_features_arr = None, conv_strides = [1, 1, 1, 1],\
                             pool_size = [1, 1, 1, 1], pool_type = 'max_pool', activate_type = 'ReLU',\
                             layer_norm = None):
        for i in range(n_cnn_layers):
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
                                      conv_strides = conv_strides, pool_size = pool_size, \
                                      pool_type = pool_type, activate_type = activate_type,
                                      layer_norm = layer_norm)
        return yout



    def convolution2d_residual( self, x, cov_ker_size = (5,1), in_n_features = 1, conv_strides = [1, 1, 1, 1],\
                       pool_size = [1, 1, 1, 1], pool_type = 'None', activate_type = 'ReLU', layer_norm = 1 ):

        y_1 = self.convolution2d(x, cov_ker_size = cov_ker_size, \
                                          in_n_features = in_n_features, out_n_features = in_n_features, \
                                          conv_strides = conv_strides, pool_size = pool_size, \
                                          pool_type = pool_type, activate_type = activate_type,\
                                          layer_norm = layer_norm)
        y_2 = self.convolution2d(y_1, cov_ker_size = cov_ker_size, \
                                          in_n_features = in_n_features, out_n_features = in_n_features, \
                                          conv_strides = conv_strides, pool_size = pool_size, \
                                          pool_type = pool_type, activate_type = 'None',\
                                          layer_norm = layer_norm)
        y_res = self.activate(x + y_2, activate_type = activate_type)
        return y_res

    def multi_convolution2d_residual( self, x, n_cnn_layers = 1, cov_ker_size = (5,1), in_n_features = 1,\
                             conv_strides = [1, 1, 1, 1],\
                             pool_size = [1, 1, 1, 1], pool_type = 'None', activate_type = 'ReLU',\
                             layer_norm = 1):
        for i in range(n_cnn_layers):
            #set input output
            if i is 0:
                yin = x
            else:
                yin = yout
            # define cnn layers
            yout = self.convolution2d_residual(yin, cov_ker_size = cov_ker_size, \
                                      in_n_features = in_n_features, \
                                      conv_strides = conv_strides, pool_size = pool_size, \
                                      pool_type = pool_type, activate_type = activate_type,
                                      layer_norm = layer_norm)
        return yout


    def deconvolution2d( self, x, cov_ker_size = (5,1), in_n_features = 1,\
                          out_n_features = 1, conv_strides = [1, 1, 1, 1],\
                         out_shape = None, activate_type = 'ReLU', layer_norm = None):
        if self.debug is 1:
            x = tf.Print(x, [tf.shape(x)],
                               message='Shape of input in deconv ',
                               summarize=4, first_n=1)
        batch_size = tf.shape(x)[0]
        if out_shape is None:
            out_shape    = np.array((batch_size,1,1,1))
            out_shape[1] = conv_strides[1] * tf.shape(x)[1]#int(x.shape[1])
            out_shape[2] = conv_strides[2] * tf.shape(x)[2]#int(x.shape[2])
            out_shape[3] = out_n_features
            out_shape    = tuple(out_shape)

        #weighting and bias
        W_deconv = self.weight_variable([cov_ker_size[0], cov_ker_size[1],\
                                         out_n_features, in_n_features])
        #f_shape = (cov_ker_size[0], cov_ker_size[1], 1, in_n_features)
        #W_deconv = self.get_deconv_filter(f_shape)
        #W_deconv = np.ones((cov_ker_size[0], cov_ker_size[1], in_n_features, in_n_features),dtype = np.float32)
        b_deconv = self.bias_variable([out_n_features])
        h_deconv = self.deconv2d(x, W_deconv, out_shape = out_shape, \
                               strides = conv_strides) + b_deconv


        if layer_norm is not None:
            h_ln = tf.contrib.layers.layer_norm(h_deconv, center=True, scale=True)
        else:
            h_ln = h_deconv
        y_act    = self.activate( h_ln, activate_type = activate_type)
        if self.debug is 1:
            y_act = tf.Print(y_act, [tf.shape(x)],
                               message='Shape of output in deconv ',
                               summarize=4, first_n=1)
        return y_act

    def multi_deconvolution2d(self, x, n_cnn_layers = 1, cov_ker_size = (5,1), in_n_features_arr = None,\
                              out_n_features_arr = None, conv_strides = [1, 1, 1, 1],\
                              out_shape = None, activate_type = 'ReLU', layer_norm = None):
        for i in range(n_cnn_layers):
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
                   conv_strides  = conv_strides, out_shape = None,\
                   activate_type = activate_type, layer_norm = layer_norm)
        return yout

    def crop_or_pad( self, x, new_height, new_width ):
        old_height = x.get_shape().as_list()[1]
        old_width  = x.get_shape().as_list()[2]
        if old_height < new_height: #pad x
            h_pad_pts       = new_height - old_height #number of points to pad
            h_pad_1half_pts = h_pad_pts//2 #first half of pad points
            h_pad_2half_pts = h_pad_pts - h_pad_pts//2#second half
            x_1 = tf.pad(x,[[0, 0], [h_pad_1half_pts, h_pad_2half_pts],[0, 0], [0, 0]],"CONSTANT")#do zero padding
        elif old_height > new_height:#truncate
            h_cut_pts       = old_height - new_height#number of points to be truncated
            h_cut_1half_pts = h_cut_pts//2 #first half of points to be truncated
            h_cut_2half_pts = h_cut_pts - h_cut_pts//2 #second half
            _, x_1, _ = tf.split(x, [h_cut_1half_pts, new_height, h_cut_2half_pts], 1)# split along dim 1
        else:
            x_1 = x #do nothing if size equal
        if old_width < new_width: #pad x
            w_pad_pts       = new_width - old_width #number of points to pad
            w_pad_1half_pts = w_pad_pts//2 #first half of pad points
            w_pad_2half_pts = w_pad_pts - w_pad_pts//2#second half
            x_2 = tf.pad(x_1,[[0, 0],[0, 0], [w_pad_1half_pts, w_pad_2half_pts], [0, 0]],"CONSTANT")#do zero padding
        elif old_width > new_width:#truncate
            w_cut_pts       = old_width - new_width#number of points to be truncated
            w_cut_1half_pts = w_cut_pts//2 #first half of points to be truncated
            w_cut_2half_pts = w_cut_pts - w_cut_pts//2 #second half
            _, x_2, _ = tf.split(x_1, [w_cut_1half_pts, new_width, w_cut_2half_pts], 2)# split along dim 1
        else:
            x_2 = x_1 #do nothing if size equal
        return x_2

    def merge( self, x1, x2, axis, merge_type = 'concat', resize_x2 = 0 ):
        if resize_x2 is 1: #resize x2 based on the high/width size of x1
            x2 = self.crop_or_pad(x2, x1.get_shape().as_list()[1], x1.get_shape().as_list()[2])#or try .get_shape().as_list()
        if merge_type is 'add':
            y = tf.add(x1, x2)
        elif merge_type is 'multiply': #element wise multiply
            y = tf.multiply(x1, x2)
        elif merge_type is 'concat': #concatenate
            y = tf.concat([x1, x2],axis)
        elif merge_type is 'max': #element wise max
            y = tf.maximum(x1, x2)
        elif merge_type is 'min': #element wise min
            y = tf.minimum(x1, x2)
        else:
            y = x1
        return y
