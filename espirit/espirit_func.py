import numpy as np
import scipy.io as sio
import pics.proximal_func as pf
import pics.CS_MRI_solvers_func as solvers
#import tvop as tv
import utilities.utilities_func as ut
import pics.operators_class as op
import pics.hankel_func as hk
import utilities.utilities_class as utc
import scipy
from   nufft.nufft_func import nufftfreqs2d, nufftfreqs2d, nufftfreqs3d
import numpy as np
import scipy.signal as ss
from sklearn.utils.extmath import randomized_svd
"""
def array_outer_product(A, B, result=None):
    ''' Compute the outer-product in the final two dimensions of the given arrays.
    If the result array is provided, the results are written into it.
    '''
    assert(A.shape[:-1] == B.shape[:-1])
    if result is None:
        result=np.zeros(A.shape+B.shape[-1:], dtype=A.dtype)
    if A.ndim==1:
        result[:,:]=np.outer(A, B)
    else:
        for idx in range(A.shape[0]):
            array_outer_product(A[idx,...], B[idx,...], result[idx,...])
    return result
"""
def hanning2d( a, b ):
    # build 2d window
    w2d = np.outer(ss.hanning(a), ss.hanning(b))
    return np.sqrt(w2d)

def hanning3d( a, b, c ):
    w2d = np.outer(ss.hanning(a), ss.hanning(b))
    w3d = np.zeros((a,b,c))
    for i in range(a):
        w3d[i, :, :] = np.outer(w2d[i,:].flatten(), ss.hanning(c))
    return w3d**(1.0/3.0)

def hamming2d( a, b ):
    # build 2d window
    w2d = np.outer(ss.hamming(a), ss.hamming(b))
    return np.sqrt(w2d)

def hamming3d( a, b, c ):
    w2d = np.outer(ss.hamming(a), ss.hamming(b))
    w3d = np.zeros((a,b,c))
    for i in range(a):
        w3d[i, :, :] = np.outer(w2d[i,:].flatten(), ss.hamming(c))
    return w3d**(1.0/3.0)


#direct 2d FT, transfer k-space coefficients (k_c, array) to i point in image space at (ix,iy)
# ix, iy the position in image space
#kx, ky the grid in k-space
def dft2d_im1point( ix, iy, k_c, df=1.0, iflag=1 ):
    k_shape   = k_c.shape
    sign      = -1 if iflag < 0 else 1
    #grid in k-space
    kx, ky    = nufftfreqs2d(k_shape[0], k_shape[1], df)
    #kx, ky    = np.mgrid[0:k_shape[0], 0:k_shape[1]]
    kx = 1.0*kx/(k_shape[0])
    ky = 1.0*ky/(k_shape[1])
    k_i_prod  = np.multiply(k_c, np.exp(sign * 1j * 2.0 * np.pi * (\
        np.multiply(kx, ix) + np.multiply(ky, iy))))
    return (1.0/np.prod(k_shape[0:2])) * np.sum(k_i_prod)

#2d dft return image im, this function is design for testing purpose
def dft2d_warp( i_ms, i_mt, k_c, df=1.0, iflag=1 ):
    array_x = np.mgrid[-(i_ms//2):i_ms-(i_ms//2)]/(i_ms*1.0/k_c.shape[0])
    array_y = np.mgrid[-(i_mt//2):i_mt-(i_mt//2)]/(i_mt*1.0/k_c.shape[1])
    im      = np.zeros((i_ms,i_mt),dtype=k_c.dtype)
    for idx_x in range(i_ms):
        for idx_y in range(i_mt):
            im[idx_x,idx_y] = dft2d_im1point(array_x[idx_x], array_y[idx_y], k_c)
    return im

def test1():
    N         = 20
    k1, k2    = nufftfreqs2d(N, N)
    k_c       = np.cos(np.multiply(5,k1))+1j*np.sin(np.multiply(5,k2))#
    #k_c       = np.ones((N,N))#    
    hwin      = hamming2d(N,N)
    #ut.plotgray(np.absolute(hwin)) 
    # apply hamming window   
    k_c       = np.multiply(k_c, hwin)
    im        = dft2d_warp(N, N, k_c)
    ut.plotim1(np.absolute(im), bar = True)
    #use std fft lib
    ft        = op.FFT2d()
    npim      = ft.backward(k_c)
    ut.plotim1(np.absolute(npim), bar = True)
    ut.plotim1(np.absolute(im-npim), bar = True)
    #interpolatation
    im_int    = dft2d_warp(5*N, 5*N, k_c)
    ut.plotim1(np.absolute(im_int), bar = True)    


#direct 3d FT, transfer k-space coefficients (k_c, array) to i point in image space at (ix,iy)
# ix, iy, iz the position in image space
#kx, ky, kz the grid in k-space
def dft3d_im1point( ix, iy, iz, k_c, df=1.0, iflag=1 ):
    k_shape    = k_c.shape
    sign       = -1 if iflag < 0 else 1
    kx, ky, kz = nufftfreqs3d(k_shape[0], k_shape[1], k_shape[2], df)
    kx = 1.0*kx/(k_shape[0])
    ky = 1.0*ky/(k_shape[1])
    kz = 1.0*kz/(k_shape[2])
    k_i_prod   = np.multiply(k_c, np.exp(sign * 1j * 2.0 * np.pi * (\
        np.multiply(kx, ix) + np.multiply(ky, iy) + np.multiply(kz, iz))))
    
    return (1.0/np.prod(k_shape[0:3])) * sum(k_i_prod.flatten())#

#3d dft return image im, this function is design for testing purpose
def dft3d_warp( i_ms, i_mt, i_mu, k_c, df=1.0, iflag=1 ):
    array_x = np.mgrid[-(i_ms//2):i_ms-(i_ms//2)]/(i_ms*1.0/k_c.shape[0])
    array_y = np.mgrid[-(i_mt//2):i_mt-(i_mt//2)]/(i_mt*1.0/k_c.shape[1])
    array_z = np.mgrid[-(i_mu//2):i_mu-(i_mu//2)]/(i_mu*1.0/k_c.shape[2])
    im      = np.zeros((i_ms,i_mt,i_mu),dtype=k_c.dtype)
    for idx_x in range(i_ms):
        for idx_y in range(i_mt):
            for idx_z in range(i_mu):
                im[idx_x,idx_y,idx_z]\
                = dft3d_im1point(array_x[idx_x], array_y[idx_y], array_z[idx_z], k_c)
    return im

def test2():
    N         = 20
    k1, k2, k3 = nufftfreqs3d(N, N, N)
    k_c        = np.cos(np.multiply(5,k1))+1j*np.sin(np.multiply(5,k2))#
    #k_c       = np.ones((N,N,N))#    
    hwin       = hamming3d(N,N,N)
    ut.plotim3(np.absolute(hwin),[4,-1]) 
    # apply hamming window   
    k_c       = np.multiply(k_c, hwin)
    im        = dft3d_warp(N, N, N, k_c)
    ut.plotim3(np.absolute(im),[4,-1], bar = True)
    #use std fft lib
    ft        = op.FFTnd()
    npim      = ft.backward(k_c)
    ut.plotim3(np.absolute(npim),[4,-1], bar = True)
    ut.plotim3(np.absolute(im-npim),[4,-1], bar = True)
    #interpolatation
    im_int    = dft3d_warp(2*N, 2*N, N, k_c)
    ut.plotim3(np.absolute(im_int),[4,-1], bar = True) 
"""
2d espirit
inputs
xcrop is 3d matrix with first two dimentions as nx,ny and third one as coil
nsingularv = 150, number of truncated singular vectors
outputs
Vim the sensitivity map
sim the singular value map

"""
def espirit_2d( xcrop, x_shape, nsingularv = 150, hkwin_shape = (16,16), pad_before_espirit = 0, pad_fact = 1 ):
    ft = op.FFT2d()#2d fft operator
    #timing = utc.timing()
    #multidimention tensor as the block hankel matrix
    #first 2 are x, y dims with rolling window size of hkwin_shape
    #last 1 is coil dimension, with stride of 1
    #timing.start()
    h = hk.hankelnd_r(xcrop, (hkwin_shape[0], hkwin_shape[1], 1))
    #timing.stop().display('Create Hankel ').start()
    dimh = h.shape
    #flatten the tensor to create a matrix= [flatten(fist3 dims), flatten(last3 dims)]
    #the second dim of hmtx contain coil information, i.e. dimh[2]=1, dimh[5]=N_coils    
    hmtx = h.reshape(( dimh[0]* dimh[1]* dimh[2], dimh[3], dimh[4], dimh[5])).\
             reshape(( dimh[0]* dimh[1]* dimh[2], dimh[3]* dimh[4]* dimh[5]))
    #timing.stop().display('Reshape Hankel ').start()
    #svd, could try other approaches
    # V has the coil information since the second dim of hmtx has coil data    
    #U, s, V = np.linalg.svd(hmtx, full_matrices=False)
    U, s, V = scipy.sparse.linalg.svds(hmtx, nsingularv )   
    #U, s, V = scipy.sparse.linalg.svds(hmtx, nsingularv )
    #timing.stop().display('SVD ')
    #S = np.diag(s)
    #ut.plotim1(np.absolute(V[:,0:150]).T)#plot V singular vectors
    #ut.plot(s)#plot singular values
    #invh = np.zeros(x.shape,complex)
    #print h.shape
    #hk.invhankelnd(h,invh,(2,3,1))
    
    #reshape vn to generate k-space vn tensor
    vn = V[0:nsingularv,:].reshape((nsingularv,dimh[3],dimh[4],dimh[5])).transpose((1,2,0,3))

    #zero pad vn, vn matrix of reshaped singular vectors,
    #dims of vn: nx,ny,nsingularv,ncoil
    #nx = x_shape[0]
    #ny = x_shape[1]

    # do pading before espirit
    if pad_before_espirit is 0:
        nx = min(pad_fact * xcrop.shape[0], x_shape[0])
        ny = min(pad_fact * xcrop.shape[1], x_shape[1])  
    else:
        nx = x_shape[0]
        ny = x_shape[1]      

    nc = x_shape[2]

    hwin       = hamming2d(vn.shape[0],vn.shape[1])
    # apply hamming window   
    vn       = np.multiply(vn, hwin[:,:,np.newaxis,np.newaxis])
 
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
    
    if pad_before_espirit is 0:
        Vim = ft.backward(ut.pad2d(ft.forward(Vim),x_shape[0],x_shape[1]))
        sim = ft.backward(ut.pad2d(ft.forward(sim),x_shape[0],x_shape[1]))
   
    #plot first eigen vector, eigen value
    ut.plotim3(np.absolute(Vim))
    ut.plotim1(np.absolute(sim))
    #Vim_dims_name = ['x', 'y', 'coil']
    #sim_dims_name = ['x', 'y']
    return Vim, sim #, Vim_dims_name, sim_dims_name

"""
3d espirit
inputs
xcrop is 3d matrix with first two dimentions as nx,ny and third one as coil
nsingularv = 150, number of truncated singular vectors
outputs
Vim the sensitivity map
sim the singular value map

"""
def espirit_3d( xcrop, x_shape, nsingularv = 150, hkwin_shape = (16,16,16), pad_before_espirit = 0, pad_fact = 1 ):
    ft = op.FFTnd((0,1,2))#3d fft operator
    timing = utc.timing()
    #multidimention tensor as the block hankel matrix
    #first 2 are x, y dims with rolling window size of hkwin_shape
    #last 1 is coil dimension, with stride of 1    
    # output dims are : (3_hankel_dims + 1_coil_dim)_win_size + (3_hankel_dims + 1_coil_dim)_rolling_times
    timing.start()
    h = hk.hankelnd_r(xcrop, (hkwin_shape[0], hkwin_shape[1], hkwin_shape[2], 1))
    timing.stop().display('Create Hankel ').start()
    dimh = h.shape
    #flatten the tensor to create a matrix= [flatten(fist4 dims), flatten(last4 dims)]
    #the second dim of hmtx contain coil information, i.e. dimh[3]=1, dimh[7]=N_coils
    hmtx = h.reshape(( dimh[0]* dimh[1]* dimh[2]* dimh[3], dimh[4], dimh[5], dimh[6], dimh[7])).\
              reshape((dimh[0]* dimh[1]* dimh[2]* dimh[3], dimh[4]* dimh[5]* dimh[6]* dimh[7]))
    timing.stop().display('Reshape Hankel ').start()
    #svd, could try other approaches
    # V has the coil information since the second dim of hmtx has coil data
    
    U, s, V = np.linalg.svd(hmtx, full_matrices=False)
    #U, s, V = scipy.sparse.linalg.svds(hmtx, nsingularv )
    #U, s, V = randomized_svd(hmtx, n_components=nsingularv,n_iter=5,random_state=None)
    timing.stop().display('SVD ').start()

    #S = np.diag(s)
    #ut.plotim1(np.absolute(V[:,0:150]).T)#plot V singular vectors
    #ut.plot(s)#plot singular values
    #invh = np.zeros(x.shape,complex)
    #print h.shape
    #hk.invhankelnd(h,invh,(2,3,1))

    #reshape vn to generate k-space vn tensor
    #first dim is singular vector, which is transposed to the second last dimension
    vn = V[0:nsingularv,:].reshape((nsingularv,dimh[4],dimh[5],dimh[6],dimh[7])).transpose((1,2,3,0,4))

    #zero pad vn, vn matrix of reshaped singular vectors,
    #dims of vn: nx,ny,nsingularv,ncoil
    #do pading before espirit, reduce the memory requirement
    if pad_before_espirit is 0:
        nx = min(pad_fact * xcrop.shape[0], x_shape[0])
        ny = min(pad_fact * xcrop.shape[1], x_shape[1])  
        nz = min(pad_fact * xcrop.shape[2], x_shape[2])
    else:
        nx = x_shape[0]
        ny = x_shape[1]
        nz = x_shape[2]  

    nc = x_shape[3]

    hwin       = hamming3d(vn.shape[0],vn.shape[1],vn.shape[2])
    # apply hamming window   
    vn       = np.multiply(vn, hwin[:,:,:,np.newaxis,np.newaxis])

    vn = ut.pad3d(vn,nx,ny,nz)
    #plot first singular vecctor Vn[0]
    imvn = ft.backward(vn)
    #ut.plotim3(np.absolute(imvn[:,:,0,:].squeeze()))#spatial feature of V[:,1] singular vector
    sim = np.zeros((nx,ny,nz),dtype=vn.dtype)
    Vim = np.zeros((nx,ny,nz,nc),dtype=vn.dtype)

    for ix in range(nx):
        for iy in range(ny):
            for iz in range (nz):
                vpix    = imvn[ix,iy,iz,:,:].squeeze()
                vpix    = np.matrix(vpix).transpose()
                vvH     = vpix.dot(vpix.getH())
                U, s, V = np.linalg.svd(vvH, full_matrices=False)
                #s, V = numpy.linalg.eig(vvH)
                print(V)
                print(s)
                sim[ix,iy,iz]   = s[0]
                Vim[ix,iy,iz,:] = V[0,:].squeeze()
    
    timing.stop().display('ESPIRIT ')
    #pad the image after espirit
    if pad_before_espirit is 0:
        Vim = ft.backward(ut.pad3d(ft.forward(Vim),x_shape[0],x_shape[1],x_shape[2]))
        sim = ft.backward(ut.pad3d(ft.forward(sim),x_shape[0],x_shape[1],x_shape[2]))


    #plot first eigen vector, which is coil sensitvity map, and eigen value
    ut.plotim3(np.absolute(Vim[Vim.shape[0]//2,:,:,:].squeeze()))
    ut.plotim1(np.absolute(sim[Vim.shape[0]//2,:,:].squeeze()))
    #Vim_dims_name = ['x', 'y', 'z', 'coil']
    #sim_dims_name = ['x', 'y', 'z']
    return Vim, sim #, Vim_dims_name, sim_dims_name

