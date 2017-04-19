import numpy as np
"""
hankel1d

from poster
http://stackoverflow.com/questions/4923617/efficient-numpy-2d-array-construction-from-1d-array
Make an ndarray with a rolling window of the second dimension

Parameters
----------
a : array input
win_w : int Size of rolling window
step_w : step size for moving rolling window each time
Returns
-------
2d Hankel matrix with two dimension: (a_w - win_w)/step_w + 1 and win_w.

usage:
--------
a = np.arange(10)
print hankel1d(a, 3)
"""

def hankel1d(a, win_w, step_w = 1):
    #reshape of raw hankel tensor with
    #dims   : times_of_moving, win_w
    bh_shape = ((a.size - win_w)//step_w + 1, win_w)
    #strides define the moving steps along each dim of the block hankel matrix
    #strides: step_of__moving, step_1_each_item
    bh_strides = (step_w*a.itemsize, a.itemsize)

    return np.lib.stride_tricks.as_strided(a, shape=bh_shape, strides=bh_strides)

def invhankel1d(h, a, win_w, step_w = 1):

    #reshape of raw hankel tensor with
    #dims   : times_of_moving, win_w
    bh_shape = ((a.size - win_w)//step_w + 1, win_w)
    #strides define the moving steps along each dim of the block hankel matrix
    #strides: step_of__moving, step_1_each_item
    bh_strides = (step_w*a.itemsize, a.itemsize)
    print bh_shape, bh_strides
    np.lib.stride_tricks.as_strided(a, shape=bh_shape, strides=bh_strides)[:] = h
    return a

#window dimentions first reverse the order of window dims and rolling/repeating dims in output
def hankel1d_r(a, win_w, step_w = 1):
    return hankel1d(a, win_w, step_w).transpose((1,0))

"""
hankel2d
Parameters
----------
a : 2d_matrix input

win_h : int, height of rolling window, the window size of second dimention
win_w : int, weight of rolling window, the window size of third dimention

step_h : int, step size along height for moving the rolling window, the strides of moving window along second dimention
step_w : int, step size along weight for moving the rolling window, the strides of moving window along second dimention


Returns
-------
4d block Hankel matrix with dimensions of bh_shape.

Usage:
--------
a = np.arange(36).reshape(6,6)
print a
print hankel2d (a,3,3,1,1)
"""

def hankel2d(a,win_h,win_w,step_h=1,step_w=1):
    # get the shape of input matrix: h, height; w, weight
    h,w = a.shape
    # define the shape of block hankel matrix
    # dims          : num_moving_h,  num_moving_w, win_size_h , win_size_w
    bh_shape = ( ((h-win_h)//step_h + 1)  , ((w-win_w)//step_w + 1) , win_h, win_w )
    # each move/stride step: step_h * w  ,  step_w      , w          , 1
    bh_strides = ( step_h * w * a.itemsize, step_w * a.itemsize, w * a.itemsize, a.itemsize )

    return np.lib.stride_tricks.as_strided(a, shape=bh_shape, strides=bh_strides)

#reverse the order of dims in output, i.e. the Hankel window dims first
def hankel2d_r(a,win_h,win_w,step_h,step_w):
    return hankel2d(a,win_h,win_w,step_h,step_w).transpose((2,3,0,1))

"""
hankel3d
3d slideing window
Parameters
----------
a : 3d_matrix input
win_l : int, length of rolling window, the window size of first dimention
win_h : int, height of rolling window, the window size of second dimention
win_w : int, weight of rolling window, the window size of third dimention

step_l : int, step size along length for moving the rolling window, the strides of moving window along first dimention
step_h : int, step size along height for moving the rolling window, the strides of moving window along second dimention
step_w : int, step size along weight for moving the rolling window, the strides of moving window along third dimention

Returns
-------
block Hankel matrix with dimensions of bh_shape.

Usage:
--------
"""
def hankel3d( a, win_l, win_h, win_w, step_l=1, step_h=1, step_w=1 ):
    # get the shape of input matrix: h, height; w, weight; l, length
    l,h,w = a.shape
    # define shape of block hankel matrix
    # dims          : num_moving_l, num_moving_h,  num_moving_w, win_size_l , win_size_h , win_size_w
    bh_shape = ( ((l-win_l)//step_l + 1), ((h-win_h)//step_h + 1)  , ((w-win_w)//step_w + 1), win_l, win_h, win_w )
    # each move step: step* w*h   , step_h * w  ,  step_w      ,  w*h       , w          , 1
    bh_strides = ( step_l * w * h * a.itemsize, step_h * w * a.itemsize, step_w * a.itemsize, w * h * a.itemsize, w * a.itemsize, a.itemsize )

    return np.lib.stride_tricks.as_strided(a, shape=bh_shape, strides=bh_strides)

#window dimentions first, reverse the order of window dims and rolling/repeating dims in output
def hankel3d_r(a,win_l,win_h,win_w,step_l,step_h,step_w):
    return hankel3d(a,win_l,win_h,win_w,step_l,step_h,step_w).transpose((3,4,5,0,1,2))


"""
more general hankel function

Parameters
----------
a : ndim tensor
win_shape : the size of hankel matrix, match the dim with a
win_strides : int, strides for moving the rolling window, default should be all one

Returns
-------
block Hankel matrix with dimensions two times larger than the input.
the first half of dimentions are the movements of window,
the second half of dimentions are the window

Usage:
--------
a = np.arange(4*5*6).reshape(4,5,6)
print a
print hankelnd(a, (2, 3, 4)
"""
def hankelnd( a, win_shape, win_strides = None ):
    if win_strides is None:
        win_strides = np.ones(win_shape.__len__()).astype(int)

    win_shape = np.array(win_shape)
    win_strides = np.array(win_strides)
    #print win_strides,win_shape,a.shape

    #get size and shape
    a_shape = np.array(a.shape)
    a_strides = np.array(a.strides)
    #define shape of block hankel matrix, half is num_movements of hankel matrix, half is num_movements inside window
    bh_shape = np.concatenate((np.divide(a_shape - win_shape, win_strides).astype(int) + 1,win_shape))
    #define each move/strides in a hankel window, match bh_shape
    # half is strides of hankel matrix, half is strides of movements inside window
    bh_strides = np.concatenate((np.multiply(win_strides,a_strides),a_strides))

    return np.lib.stride_tricks.as_strided(a, shape=bh_shape, strides=bh_strides)

#invert hankel operator, input is h, hankel matrix, output is original data
def invhankelnd( h, a, win_shape, win_strides = None ):
    if win_strides is None:
        win_strides = np.ones(win_shape.__len__()).astype(int)

    win_shape = np.array(win_shape)
    win_strides = np.array(win_strides)
    #print win_strides,win_shape,a.shape

    #get size and shape
    a_shape = np.array(a.shape)
    a_strides = np.array(a.strides)
    #define shape of block hankel matrix, half is num_movements of hankel matrix, half is num_movements inside window
    bh_shape = np.concatenate((np.divide(a_shape - win_shape, win_strides).astype(int) + 1,win_shape))
    #define each move/strides in a hankel window, match bh_shape
    # half is strides of hankel matrix, half is strides of movements inside window
    bh_strides = np.concatenate((np.multiply(win_strides,a_strides),a_strides))

    np.lib.stride_tricks.as_strided(a, shape=bh_shape, strides=bh_strides)[:] = h
    return a


# the first half dimentions are window dimentions,
# the second half dimentions are rolling/repeating dimentions
def hankelnd_r( a, win_shape, win_strides = None ):
    if win_strides is None:
        win_strides = np.ones(win_shape.__len__()).astype(int)

    win_shape = np.array(win_shape)
    win_strides = np.array(win_strides)
    #print win_strides,win_shape,a.shape

    #get size and shape
    a_shape = np.array(a.shape)
    a_strides = np.array(a.strides)
    #define shape of block hankel matrix, half is num_movements of hankel matrix, half is num_movements inside window
    bh_shape = np.concatenate((win_shape, np.divide(a_shape - win_shape, win_strides).astype(int) + 1))
    #define each move/strides in a hankel window, match bh_shape
    # half is strides of hankel matrix, half is strides of movements inside window
    bh_strides = np.concatenate((a_strides, np.multiply(win_strides,a_strides)))

    return np.lib.stride_tricks.as_strided(a, shape=bh_shape, strides=bh_strides)

"""
test function
"""
def test():
    #test hankel1d
    #a = np.arange(10)
    #print a
    #print hankel1d(a, 3)

    #invert hankel1d
    #h = hankel1d(a, 3)
    #invh = np.zeros(10)
    #print invhankel1d(h, invh, 3)

    #test hankel2d
    #a = np.arange(48).reshape(6,8)
    #print a
    #print hankel2d (a,3,4,3,2)

    #test hankel3d
    #a = np.arange(4*5*6).reshape(4,5,6)
    #print a
    #print hankel3d(a, 2, 3, 4)

    #test hankelnd
    a = np.arange(4*5*6).reshape(4,5,6)
    print a
    h = hankelnd(a, (2, 3, 4))
    invh = np.zeros(a.shape)
    print invhankelnd(h,invh,(2,3,4))

#if __name__ == "__main__":
    #test()
