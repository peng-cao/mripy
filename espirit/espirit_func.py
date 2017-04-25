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
    
    #U, s, V = np.linalg.svd(hmtx, full_matrices=False)
    U, s, V = scipy.sparse.linalg.svds(hmtx, nsingularv )

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
                #U, s, V = np.linalg.svd(vvH, full_matrices=False)
                s, V = numpy.linalg.eig(vvH)
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
