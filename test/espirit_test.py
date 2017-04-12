import numpy as np
import scipy.io as sio
import proximal_func as pf
import CS_MRI_solvers as solvers
import tvop as tv
import utilities_func as ut
import operators_class as op
import hankel_func as hk

import matplotlib.pyplot as plt
from matplotlib.mlab import PCA


ft = op.FFT2d()

# simulated image
mat_contents = sio.loadmat('/home/pcao/matlab/espirit_code/data/brain_8ch.mat');
x = mat_contents["DATA"]

#ut.plotim1(np.absolute(x[:,:,0]))

im = ft.backward(x[:,:,:])
ut.plotim3(np.absolute(im))

#shape of x
nx,ny,nc = x.shape

xcrop = ut.crop2d( x, 30 )

#ksp = ft.forward(im)
#ut.plotim1(np.absolute(ksp[:,:]))

#multidimention tensor as the block hankel matrix 
h = hk.hankelnd_r(xcrop, (16, 16, 1))
dimh = h.shape
#flatten the tensor to create a matrix
hmtx = h.reshape(( dimh[0]*dimh[1]*dimh[2], dimh[3], dimh[4], dimh[5] )).reshape((dimh[0]*dimh[1]*dimh[2],dimh[3]*dimh[4]*dimh[5]))

#svd, could try other approaches
U, s, V = np.linalg.svd(hmtx, full_matrices=False)
#S = np.diag(s)
#ut.plotim1(np.absolute(V[:,0:150]).T)#plot V singular vectors
ut.plot(s)#plot sigular values
#invh = np.zeros(x.shape,complex)
#print h.shape
#hk.invhankelnd(h,invh,(2,3,1))

#reshape vn to generate k-space vn tensor
nsingular = 150#number of truncated sigular vectors
vn = V[0:nsingular,:].reshape((nsingular,dimh[3],dimh[4],dimh[5])).transpose((1,2,0,3))

#zero pad vn, vn matrix of reshaped singular vectors,
#dims of vn: nx,ny,nsingular,ncoil
vn = ut.pad2d(vn,nx,ny)
#plot first singular vecctor Vn[0]
imvn = ft.backward(vn)
#ut.plotim3(np.absolute(imvn[:,:,0,:].squeeze()))#spatial feature of V[:,1] singular vector
sim = 1j*np.zeros((nx,ny))
Vim = 1j*np.zeros((nx,ny,nc))
#Uim = 1j*np.zeros((nx,ny,nc))

for ix in range(nx):
    for iy in range(ny):
        vpix = imvn[ix,iy,:,:].squeeze()
        vpix = np.matrix(vpix).transpose()
        vvH = vpix.dot(vpix.getH())
        U, s, V = np.linalg.svd(vvH, full_matrices=False)
        sim[ix,iy] = s[0]
        Vim[ix,iy,:] = V[0,:].squeeze()
        #Uim[ix,iy,:] = U[:,0].squeeze()

#plot first eigen vector, eigen value
ut.plotim3(np.absolute(Vim))
#ut.plotim3(np.absolute(Uim))
ut.plotim1(np.absolute(sim))

