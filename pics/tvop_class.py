import numpy as np
"""
define total variation gradient and divergense functions
this class is for 2d tv minimization
prox function is from
Chambolle, An algorithm for total variation minimizations and applications, 2004
and a pdf file
Total Variation Regularization with Chambolle Algorihtm.pdf
"""
class TV2d:
    "this define functions related to totalvariation minimization"
    def __init__(self):
        self.ndim    = 2         #number of image dimension
        #self.imdim   = (0, 1)    #image dimension
        #self.tvopdim = (0, 1)    #apply tv on those dimensions
        #self.tmpdim  = self.imdim + (self.ndim,) #tmp data dimensions, which has one more dim for saving the gradient
        #self.imshape = None

    def grad( self, x ): #gradient of x
        sx, sy = x.shape
        Dx = x[np.r_[1:sx, sx-1],:] - x
        Dy = x[:,np.r_[1:sy, sy-1]] - x
        #res = np.concatenate((Dx,Dy),2)
        res = (1j)*np.zeros((sx,sy,2))
        res[:,:,0] = Dx
        res[:,:,1] = Dy
        return res

    def adjDy( self, x ): #used in computing divergense of x
        sx,sy = x.shape
        res = x[:,np.r_[0, 0:sy-1]] - x
        res[:,0] = -x[:,0]
        res[:,-1] = x[:,-2]
        return res

    def adjDx( self, x ): #used in computing divergense of x
        sx,sy = x.shape
        res = x[np.r_[0, 0:sx-1],:] - x
        res[0,:] = -x[0,:]
        res[-1,:] = x[-2,:]
        return res

    def Div( self, y ):  #divergense of x
        #res = np.zeros(y.shape)
        res = self.adjDx(y[:,:,0]) + self.adjDy(y[:,:,1])
        return res

    def amp(self, grad):
        amp = np.sqrt(np.sum(grad ** 2,axis=self.ndim))#nomalize u along the third dimension
        d = np.tile(amp[:,:,np.newaxis], (1,1,self.ndim))#.reshape(sizeg)
        return d


# this define the 3d tv operator including gradient and divergense functions
class TV3d:
    "this define functions related to totalvariation minimization"
    def __init__(self):
        self.ndim    = 3         #number of image dimension
        #self.imdim   = (0, 1, 2) #image dimension
        #self.tvopdim = (0, 1, 2) #apply tv on those dimensions
        #self.tmpdim  = self.imdim + (self.ndim,) #tmp data dimensions, which has one more dim for saving the gradient
        #self.imshape = None

    def grad( self, x ):  #gradient of x
        #if self.imshape is None:
        #    self.imshape = x.shape
        sx, sy, sz = x.shape
        Dx = x[np.r_[1:sx, sx-1],:,:] - x
        Dy = x[:,np.r_[1:sy, sy-1],:] - x
        Dz = x[:,:,np.r_[1:sz, sz-1]] - x
        res = np.zeros((sx,sy,sz) + (self.ndim,), dtype = x.dtype)
        res[:,:,:,0] = Dx
        res[:,:,:,1] = Dy
        res[:,:,:,2] = Dz
        return res

    def adjDx( self, x ):  #used in computing divergense of x
        sx, sy, sz = x.shape
        res = x[np.r_[0, 0:sx-1],:,:] - x
        res[ 0, :, :] = -x[ 0, :, :]
        res[-1, :, :] =  x[-2, :, :]
        return res

    def adjDy( self, x ): #use
        sx, sy, sz = x.shape
        res = x[:,np.r_[0, 0:sy-1],:] - x
        res[ :, 0, :] = -x[ :, 0, :]
        res[ :,-1, :] =  x[ :,-2, :]
        return res

    def adjDz( self, x ):
        sx, sy, sz = x.shape
        res = x[:,:,np.r_[0, 0:sz-1]] - x
        res[ :, :, 0] = -x[ :, :, 0]
        res[ :, :,-1] =  x[ :, :,-2]
        return res

    def Div( self, y ):
        #res = np.zeros(y.shape)
        res = self.adjDx(y[:,:,:,0]) + self.adjDy(y[:,:,:,1]) + self.adjDz(y[:,:,:,2])
        return res

    def amp(self, grad):
        amp = np.sqrt(np.sum(grad ** 2,axis=self.ndim))#nomalize u along the third dimension
        d = np.tile(amp[:,:,:,np.newaxis], (1,1,1,self.ndim))#.reshape(sizeg)
        return d
