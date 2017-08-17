import numpy as np
"""
function computes the finite difference transform of the image

"""
def grad(x):
    sx, sy = x.shape
    Dx = x[np.r_[1:sx, sx-1],:] - x
    Dy = x[:,np.r_[1:sy, sy-1]] - x
    #res = np.concatenate((Dx,Dy),2)
    res = np.zeros((sx,sy,2), dtype = x.dtype)
    res[:,:,0] = Dx
    res[:,:,1] = Dy
    return res

def adjDy(x):
    sx,sy = x.shape
    res = x[:,np.r_[0, 0:sy-1]] - x
    res[:,0] = -x[:,0]
    res[:,-1] = x[:,-2]
    return res

def adjDx(x):
    sx,sy = x.shape
    res = x[np.r_[0, 0:sx-1],:] - x
    res[0,:] = -x[0,:]
    res[-1,:] = x[-2,:]
    return res
   
def Div(y):
    #res = np.zeros(y.shape)
    res = adjDx(y[:,:,0]) + adjDy(y[:,:,1])
    return res
