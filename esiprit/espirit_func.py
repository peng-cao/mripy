import numpy as np
import scipy.io as sio
import pics.proximal_func as pf
import pics.CS_MRI_solvers_func as solvers
#import tvop as tv
import utilities.utilities_func as ut
import pics.operators_class as op
import pics.hankel_func as hk

"""
2d espirit
inputs
xcrop is 3d matrix with first two dimentions as nx,ny and third one as coil
nsingular = 150, number of truncated sigular vectors
outputs
Vim the sensitivity map
sim the sigular value map

"""
def espirit2d( xcrop, x_shape, nsigular = 150, hkwin_shape=(16,16) ):
	ft = op.FFT2d()#2d fft operator
    #multidimention tensor as the block hankel matrix 
    h = hk.hankelnd_r(xcrop, (hkwin_shape[0], hkwin_shape[1], 1))
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
    
    vn = V[0:nsingular,:].reshape((nsingular,dimh[3],dimh[4],dimh[5])).transpose((1,2,0,3))

    #zero pad vn, vn matrix of reshaped singular vectors,
    #dims of vn: nx,ny,nsingular,ncoil
    nx = x_shape[0]
    ny = x_shape[1]
    nc = x_shape[2]
    vn = ut.pad2d(vn,nx,ny)
    #plot first singular vecctor Vn[0]
    imvn = ft.backward(vn)
    #ut.plotim3(np.absolute(imvn[:,:,0,:].squeeze()))#spatial feature of V[:,1] singular vector
    sim = 1j*np.zeros((nx,ny))
    Vim = 1j*np.zeros((nx,ny,nc))

    for ix in range(nx):
        for iy in range(ny):
            vpix = imvn[ix,iy,:,:].squeeze()
            vpix = np.matrix(vpix).transpose()
            vvH = vpix.dot(vpix.getH())
            U, s, V = np.linalg.svd(vvH, full_matrices=False)
            sim[ix,iy] = s[0]
            Vim[ix,iy,:] = V[0,:].squeeze()

    #plot first eigen vector, eigen value
    ut.plotim3(np.absolute(Vim))
    ut.plotim1(np.absolute(sim))
    return Vim, sim

"""
3d espirit
inputs
xcrop is 3d matrix with first two dimentions as nx,ny and third one as coil
nsingular = 150, number of truncated sigular vectors
outputs
Vim the sensitivity map
sim the sigular value map

"""
def espirit3d( xcrop, nsigular = 150 ,hkwin_shape=(16,16,16) ):
    ft = op.FFTnd((0,1,2))#3d fft operator
    #multidimention tensor as the block hankel matrix 
    h = hk.hankelnd_r(xcrop, (hkwin_shape[0], hkwin_shape[1], hkwin_shape[2], 1))
    dimh = h.shape
    #flatten the tensor to create a matrix
    hmtx = h.reshape(( dimh[0]*dimh[1]*dimh[2]*dimh[3], dimh[4], dimh[5], dimh[6], dimh[7] )).reshape((dimh[0]*dimh[1]*dimh[2]*dimh[3],dimh[4]*dimh[5]*dimh[6]))

    #svd, could try other approaches
    U, s, V = np.linalg.svd(hmtx, full_matrices=False)
    #S = np.diag(s)
    #ut.plotim1(np.absolute(V[:,0:150]).T)#plot V singular vectors
    ut.plot(s)#plot sigular values
    #invh = np.zeros(x.shape,complex)
    #print h.shape
    #hk.invhankelnd(h,invh,(2,3,1))

    #reshape vn to generate k-space vn tensor
    
    vn = V[0:nsingular,:].reshape((nsingular,dimh[4],dimh[5],dimh[6],dimh[7])).transpose((1,2,3,0,4))

    #zero pad vn, vn matrix of reshaped singular vectors,
    #dims of vn: nx,ny,nsingular,ncoil
    vn = ut.pad3d(vn,nx,ny,nz)
    #plot first singular vecctor Vn[0]
    imvn = ft.backward(vn)
    #ut.plotim3(np.absolute(imvn[:,:,0,:].squeeze()))#spatial feature of V[:,1] singular vector
    sim = 1j*np.zeros((nx,ny,nz))
    Vim = 1j*np.zeros((nx,ny,nz,nc))

    for ix in range(nx):
        for iy in range(ny):
        	for iz in range (nz):
                vpix = imvn[ix,iy,iz,:,:].squeeze()
                vpix = np.matrix(vpix).transpose()
                vvH = vpix.dot(vpix.getH())
                U, s, V = np.linalg.svd(vvH, full_matrices=False)
                sim[ix,iy,iz] = s[0]
                Vim[ix,iy,iz,:] = V[0,:].squeeze()

    #plot first eigen vector, eigen value
    ut.plotim3(np.absolute(Vim[:,:,1,:].squeeze()))
    ut.plotim1(np.absolute(sim[:,:,1,:].squeeze()))
    return Vim, sim
