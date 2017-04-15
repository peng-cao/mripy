import numpy as np
"""
define total variation gradient and divergense functions
this class is for 2d tv minimization 
prox function is from
Chambolle, An algorithm for total variation minimizations and applications, 2004
and a pdf file
Total Variation Regularization with Chambolle Algorihtm.pdf
"""
class totalvariation2d:
    "this define functions related to totalvariation minimization"
    def grad( self, x ):
        sx, sy = x.shape
        Dx = x[np.r_[1:sx, sx-1],:] - x
        Dy = x[:,np.r_[1:sy, sy-1]] - x
        #res = np.concatenate((Dx,Dy),2)
        res = (1j)*np.zeros((sx,sy,2))
        res[:,:,0] = Dx
        res[:,:,1] = Dy
        return res

    def adjDy( self, x ):
        sx,sy = x.shape
        res = x[:,np.r_[0, 0:sy-1]] - x
        res[:,0] = -x[:,0]
        res[:,-1] = x[:,-2]
        return res

    def adjDx( self, x ):
        sx,sy = x.shape
        res = x[np.r_[0, 0:sx-1],:] - x
        res[0,:] = -x[0,:]
        res[-1,:] = x[-2,:]
        return res
   
    def Div( self, y ):
        #res = np.zeros(y.shape)
        res = self.adjDx(y[:,:,0]) + self.adjDy(y[:,:,1])
        return res

    # this define the 3d tv operator including gradient and divergense functions
class totalvariation2d:
    "this define functions related to totalvariation minimization"
    def grad( self, x ):
        sx, sy, sz = x.shape
        Dx = x[np.r_[1:sx, sx-1],:,:] - x
        Dy = x[:,np.r_[1:sy, sy-1],:] - x
        Dz = x[:,:,np.r_[1:sz, sz-1]] - x
        #res = np.concatenate((Dx,Dy),2)
        res = (1j)*np.zeros((sx,sy,sz,2))
        res[:,:,:,0] = Dx
        res[:,:,:,1] = Dy
        res[:,:,:,2] = Dz
        return res

    def adjDx( self, x ):
        sx, sy, sz = x.shape
        res = x[np.r_[0, 0:sx-1],:,:] - x
        res[ 0, :, :] = -x[ 0, :, :]
        res[-1, :, :] =  x[-2, :, :]
        return res

    def adjDy( self, x ):
        sx, sy, sz = x.shape
        res = x[:,np.r_[0, 0:sy-1],:] - x
        res[ :, 0, :] = -x[ :, 0, :]
        res[ :,-1, :] =  x[ :,-2, :]
        return res 

    def adjDz( self, x ):
        sx, sy, sz = x.shape
        res = x[:,:,np.r_[0, 0:sy-1]] - x
        res[ :, :, 0] = -x[ :, :, 0]
        res[ :, :,-1] =  x[ :, :,-2]
        return res

    def Div( self, y ):
        #res = np.zeros(y.shape)
        res = self.adjDx(y[:,:,:,0]) + self.adjDy(y[:,:,:,1]) + self.adjDz(y[:,:,:,3])
        return res    
    
    # import tv operator
    def prox( self, y, lambda_tv, step = 0.1 ):
        #lambda_tv = 2/rho
        #nx, ny = y.shape
        sizeg = self.grad(y).shape#(nx,ny,2) #size of gradient tensor
        G = np.zeros(sizeg)#intial gradient tensor
        #G = self.initial_grad_tensor(y.shape)
        i = 0
    
        amp = lambda u : np.sqrt(np.sum(u ** 2,axis=2))#nomalize u along the third dimension

        while i < 40:
            dG = self.grad(self.Div(G)-y/lambda_tv)#gradient of G
            G = G - step*dG#gradient desent, tested to work with negative sign for gradient update
            d = np.tile(amp(G)[:,:,np.newaxis], (1,1,2))#.reshape(sizeg)
            G = G/np.maximum(d,1.0*np.ones(G.shape))#normalize to ensure the |G|<1
            i = i + 1
            #lambda_tv = lambda_tv*ntheta/np.linalg.norm(f-y)
            #print np.linalg.norm(G)
        f = y - lambda_tv * tv.Div(G)
        return f 