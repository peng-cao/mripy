import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.io as sio
from math import ceil
import time
"""
# color image plot
"""
def plotim1( im, colormap = None, title = None, bar = None, vmin = None, vmax = None, pause_close = None ):
    im = np.flip(im,0)
    fig, ax = plt.subplots()

    if colormap is None:
        cax = ax.imshow(im, cmap = cm.gray, origin='lower', interpolation='none', vmin = vmin, vmax = vmax)
    else:
        cax = ax.imshow(im, cmap = colormap, origin='lower', interpolation='none', vmin = vmin, vmax = vmax)
    ax.axis('off')
    if title is not None:
        ax.set_title(title)
    if bar is not None:
        cbar = fig.colorbar(cax)
        #cbar.ax.set_yticklabels([str(bar_ticks[0]), str(bar_ticks[-1:])])
    #plt.show()
    if pause_close is not None and pause_close > 0:
        plt.show(block = False)
        time.sleep(pause_close)
        plt.close(fig)
    else:
        plt.show()    
    return

def subplot( im1, im2, colormap = None, title = None, bar = None ):
    fig, ax = plt.subplots(1,2, figsize = (8,4))
    ax[0].imshow(im1, cmap='gray')
    ax[1].imshow(im2, cmap='gray')
    #if colormap is None:
    #    cax = ax.imshow(im, cmap = cm.gray, origin='lower', interpolation='none')
    #else:
    #    cax = ax.imshow(im, cmap = colormap, origin='lower', interpolation='none')
    #if title is not None:
    #    ax.set_title(title)
    #if bar is not None:
    #    cbar = fig.colorbar(cax)
    plt.show()
    return


"""
# color image plot, 3d input
# concatenate image along the third dim
"""
def catplotim(im, catdim = [10,-1] , colormap = None, title = None, bar = None, vmin = None, vmax = None, pause_close = None ):
    im = np.flip(im,0)
    nx,ny,nz = im.shape

    #concatenate image in 1d, along the third dim
    if catdim[0] >= nz :
        imcat = im[:,:,0]
        for i in range(nz)[0:nz-1]:
            imcat = np.concatenate([imcat, im[:,:,i+1]],1)
    else:
        # compute concatenate dim
        nz_catx = min(nz, catdim[0])
        if catdim[1] is not -1:
            nz_caty = min(np.int(ceil(1.0*nz/nz_catx)), catdim[1])
        else:
            nz_caty = np.int(ceil(1.0*nz/nz_catx))
        #intial the cat image
        imcatx = np.zeros((nx*nz_catx, ny))
        # zero pad im if nz < nz_catx * nz_caty
        if nz < nz_catx * nz_caty:
            im = np.concatenate([im, np.zeros((nx,ny,nz_catx*nz_caty-nz))], 2)
        #concatenate image in 2d, along x and y
        for j in range(nz_caty):
            imcatx = im[:,:,j*nz_catx]
            if nz_catx > 1:
                for i in range(nz_catx-1):
                    imcatx = np.concatenate([imcatx, im[:,:,i + 1 + j*nz_catx]],1)#concatenate along x
            if j is 0:
                imcat = imcatx
            else:
                imcat = np.concatenate([imcatx,imcat],0)#concatenate along y
    fig, ax = plt.subplots()
    #ax.imshow(imcat)
    if colormap is None:
        cax = ax.imshow(imcat, cmap = cm.gray, origin='lower', interpolation='none', vmin = vmin, vmax = vmax)
    else:
        cax = ax.imshow(imcat, cmap = colormap, origin='lower', interpolation='none', vmin = vmin, vmax = vmax)
    ax.axis('off')
    if title is not None:
        ax.set_title(title)
    if bar is not None:
        cbar = fig.colorbar(cax)
    #plt.show()
    if pause_close is not None and pause_close > 0:
        plt.show(block = False)
        time.sleep(pause_close)
        plt.close(fig)
    else:
        plt.show()
    return

def plotim3( im, catdim = [10,-1] , colormap = None, title = None, bar = None, vmin = None, vmax = None, pause_close = None ):
    #im = np.matrix(im)
    if len(im.shape)   == 3:
        catplotim(im, catdim = catdim , colormap = colormap, title = title, bar = bar, vmin = vmin, vmax = vmax, pause_close = pause_close)
    elif len(im.shape) == 2:
        plotim1(im, colormap = colormap, title = title, bar = bar, vmin = vmin, vmax = vmax, pause_close = pause_close)
    return

def compare_plot( im1, im2, nx, pause_close = None ):
    fig, ax = plt.subplots(nx,2, figsize = (8,4))
    for i in range(nx):
        ax[i,0].imshow(im1[i,:,:], cmap='gray', origin='lower', interpolation='none')
        ax[i,1].imshow(im2[i,:,:], cmap='gray', origin='lower', interpolation='none')
    if pause_close is not None and pause_close > 0:
        plt.show(block = False)
        time.sleep(pause_close)
        plt.close(fig)
    else:
        plt.show()

"""
#gray image plot
"""
def plotgray( im, pause_close = None ):
    im = np.flip(im,0)
    fig, ax = plt.subplots()
    ax.imshow(im, cmap=cm.gray, origin='lower', interpolation='none')
    ax.axis('off')
    #plt.show()
    if pause_close is not None and pause_close > 0:
        plt.show(block = False)
        time.sleep(pause_close)
        plt.close(fig)
    else:
        plt.show()

    return

"""
plot a line
"""
def plot( x, y=None, line_type = '-', legend = None, pause_close = None ):
    fig = plt.figure()
    # Prepare the data
    if y is None:
        y = np.linspace(0, x.size, x.size)
        plt.plot(y, x, line_type)
    else:
        plt.plot(x, y, line_type)
    # Add a legend
    #plt.legend()
    if legend is not None:
        plt.legend(legend)
    # Show the plot   
    if pause_close is not None and pause_close > 0:
        plt.show(block = False)
        time.sleep(pause_close)
        plt.close(fig)
    else:
        plt.show()


"""
load matlab mat file
"""
def loadmat( matfile, var ):
    mat_contents = sio.loadmat(matfile);
    x = mat_contents[var]
    return x

"""
#generate 2d k-space mask
nx: kx dimention size
ny: ky dimention size
center_r: full sampling center area is (2*r)**2
undersampling: undersampling rate, 1 for full sampling
"""

def mask2d( nx, ny, center_r = 15, undersampling = 0.5 ):
    #create undersampling mask
    k = int(round(nx*ny*undersampling)) #undersampling
    ri = np.random.choice(nx*ny,k,replace=False) #index for undersampling
    ma = np.zeros(nx*ny) #initialize an all zero vector
    ma[ri] = 1 #set sampled data points to 1
    mask = ma.reshape((nx,ny))

    # center k-space index range
    if center_r > 0:
        cx = np.int(nx/2)
        cy = np.int(ny/2)
        cxr = np.arange(round(cx-center_r),round(cx+center_r+1))
        cyr = np.arange(round(cy-center_r),round(cy+center_r+1))
        mask[np.ix_(map(int,cxr),map(int,cyr))] = np.ones((cxr.shape[0],cyr.shape[0])) #center k-space is fully sampled

    return mask

"""
#generate 3d k-space mask
nx: kx dimention size
ny: ky dimention size
center_r: full sampling center area is (2*r)**2
undersampling: undersampling rate, 1 for full sampling
"""

def mask3d( nx, ny, nz, center_r = [15, 15, 0], undersampling = 0.5 ):
    #create undersampling mask
    mask_shape = np.array([nx, ny, nz])
    Npts       = mask_shape.prod()#total number of data points
    k          = int(round(Npts * undersampling)) #undersampling
    ri         = np.random.choice(Npts,k,replace=False) #index for undersampling
    ma         = np.zeros(Npts) #initialize an all zero vector
    ma[ri]     = 1              #set sampled data points to 1
    mask       = ma.reshape(mask_shape)

    flag_centerfull = 1
    # x center, k-space index range
    if center_r[0] > 0:
        cxr = np.arange(-center_r[0], center_r[0] + 1) + mask_shape[0]//2
    elif center_r[0] is 0:
        cxr = np.arange(mask_shape[0])
    else:
        flag_centerfull = 0
    # y center, k-space index range
    if center_r[1] > 0:
        cyr = np.arange(-center_r[1], center_r[1] + 1) + mask_shape[1]//2
    elif center_r[1] is 0:
        cyr = np.arange(mask_shape[1])
    else:
        flag_centerfull = 0
     # z center, k-space index range
    if center_r[2] > 0:
        czr = np.arange(-center_r[2], center_r[2] + 1) + mask_shape[2]//2
    elif center_r[2] is 0:
        czr = np.arange(mask_shape[2])
    else:
        flag_centerfull = 0

    #full sampling in the center kspace
    if flag_centerfull is not 0:
        mask[np.ix_(map(int,cxr),map(int,cyr),map(int,czr))] = \
        np.ones((cxr.shape[0],cyr.shape[0],czr.shape[0])) #center k-space is fully sampled
    return mask

"""
#crop 2d k-space
nx: kx dimention size
ny: ky dimention size
center_r: full sampling center area is (2*r)**2
"""
def crop2d( data, center_r = 15 ):
    nx, ny = data.shape[0:2]
    # center k-space index range
    if center_r > 0:
        cx = np.int(nx/2)
        cy = np.int(ny/2)
        cxr = np.arange(round(cx-center_r),round(cx+center_r))
        cyr = np.arange(round(cy-center_r),round(cy+center_r))

    return data[np.ix_(map(int,cxr),map(int,cyr))]

"""
#crop 3d k-space
nx: kx dimention size
ny: ky dimention size
center_r: full sampling center area is (2*r)**2
"""
def crop3d( data, center_r = 15 ):
    nx, ny, nz = data.shape[0:3]
    # center k-space index range
    if center_r > 0:
        cx = np.int(nx/2)
        cy = np.int(ny/2)
        cz = np.int(nz/2)
        cxr = np.arange(round(cx-center_r),round(cx+center_r))
        cyr = np.arange(round(cy-center_r),round(cy+center_r))
        czr = np.arange(round(cz-center_r),round(cz+center_r))

    return data[np.ix_(map(int,cxr),map(int,cyr),map(int,czr))]

"""
zero pad the 3d k-space in kx and ky dimentions
"""
def pad3d( data, nx, ny, nz ):
    #create undersampling mask
    datsize    = data.shape
    padsize    = np.array(datsize)
    padsize[0] = nx
    padsize[1] = ny
    padsize[2] = nz
    ndata = np.zeros(tuple(padsize),dtype = data.dtype)

    # center k-space index range
    datrx = np.int(datsize[0]/2)
    datry = np.int(datsize[1]/2)
    datrz = np.int(datsize[2]/2)
    cx = np.int(nx/2)
    cy = np.int(ny/2)
    cz = np.int(nz/2)
    cxr = np.arange(round(cx-datrx),round(cx-datrx+datsize[0]))
    cyr = np.arange(round(cy-datry),round(cy-datry+datsize[1]))
    czr = np.arange(round(cz-datrz),round(cz-datrz+datsize[2]))
    #print cxr,cyr
    ndata[np.ix_(map(int,cxr),map(int,cyr),map(int,czr))] = data
    return ndata

"""
zero pad or cut the 3d k-space in kx and ky dimentions
"""
def pad_or_cut3d( data, nx, ny, nz ):
    #create undersampling mask
    datsize    = data.shape
    #ndata total size could include zeros, if do padding
    totalpadcutsize    = np.array(datsize)
    totalpadcutsize[0] = nx
    totalpadcutsize[1] = ny
    totalpadcutsize[2] = nz
    ndata = np.zeros(tuple(totalpadcutsize),dtype = data.dtype)
    # min value for data size and [nx, ny, nz], this is the output real data size (without zeros)
    minoutdatsize    = np.zeros(3)
    minoutdatsize[0] = min(datsize[0], nx)
    minoutdatsize[1] = min(datsize[1], ny)
    minoutdatsize[2] = min(datsize[2], nz)
    # half size of the out real data size
    halfrealdatx = np.int(minoutdatsize[0]/2)
    halfrealdaty = np.int(minoutdatsize[1]/2)
    halfrealdatz = np.int(minoutdatsize[2]/2)
    # index for ndata, ndata size is [nx, ny, nz, ...]
    halfx = np.int(nx/2)
    halfy = np.int(ny/2)
    halfz = np.int(nz/2)
    # left side margin stop at index: half size of ndata - half size of out_data
    # right side margin start at index: left side margin stop index + size of output data
    cxr = np.arange(round(halfx - halfrealdatx), round(halfx - halfrealdatx + minoutdatsize[0]))
    cyr = np.arange(round(halfy - halfrealdaty), round(halfy - halfrealdaty + minoutdatsize[1]))
    czr = np.arange(round(halfz - halfrealdatz), round(halfz - halfrealdatz + minoutdatsize[2]))
    # index for data, datsize
    halfx2 = np.int(datsize[0]/2)
    halfy2 = np.int(datsize[1]/2)
    halfz2 = np.int(datsize[2]/2)
    # left side margin stop at index: half size of data - half size of out_data
    # right side margin start at index: left side margin stop index + size of output data
    cxr2 = np.arange(round(halfx2 - halfrealdatx), round(halfx2 - halfrealdatx + minoutdatsize[0]))
    cyr2 = np.arange(round(halfy2 - halfrealdaty), round(halfy2 - halfrealdaty + minoutdatsize[1]))
    czr2 = np.arange(round(halfz2 - halfrealdatz), round(halfz2 - halfrealdatz + minoutdatsize[2]))
    # pad or cut data to fit in the ndata matrix, which is intialized as all zeros matrix
    ndata[np.ix_(map(int,cxr),map(int,cyr),map(int,czr))] = \
             data[np.ix_(map(int,cxr2),map(int,cyr2),map(int,czr2))]
    return ndata

"""
zero pad the 2d k-space in kx and ky dimentions
"""

def pad2d( data, nx, ny ):
    #create undersampling mask
    datsize    = data.shape
    padsize    = np.array(datsize)
    padsize[0] = nx
    padsize[1] = ny
    ndata = np.zeros(tuple(padsize),dtype = data.dtype)

    # center k-space index range
    datrx = np.int(datsize[0]/2)
    datry = np.int(datsize[1]/2)
    cx = np.int(nx/2)
    cy = np.int(ny/2)
    cxr = np.arange(round(cx-datrx),round(cx-datrx+datsize[0]))
    cyr = np.arange(round(cy-datry),round(cy-datry+datsize[1]))
    #print cxr,cyr
    ndata[np.ix_(map(int,cxr),map(int,cyr))] = data
    return ndata

"""
return the scaling of data (ksp), computed as the max in image space
"""
def optscaling( FT, b ):
    x0 = np.absolute(FT.backward(b))
    return max(x0.flatten())

"""
return the scaling of data (b)
"""
def scaling( b ):
    return max(b.flatten())

# match the dimensions of A and B, by adding 1
# A_shape and B_shape are tuples from e.g. A.shape and B.shape
def dim_match( A_shape ,B_shape ):
    #intialize A_out_shape, B_out_shape
    A_out_shape = A_shape
    B_out_shape = B_shape
    #match them by adding 1
    if   len(A_shape) < len(B_shape):
        for _ in range(len(A_shape),len(B_shape)):
            A_out_shape += (1,)
    elif len(A_shape) > len(B_shape):
        for _ in range(len(B_shape),len(A_shape)):
            B_out_shape += (1,)
    return  A_out_shape, B_out_shape

"""
calculate the root sum of square, for combining the coil elements
"""
def rss( im_coils, axis_rss=None ):
    #print(np.absolute(im_coils).flatten().max())
    #im_coils=im_coils/np.absolute(im_coils).flatten().max()
    if axis_rss is None:
        axis_rss = len(im_coils.shape) - 1 # assume the last dimention is the coil dimention
    return np.linalg.norm(im_coils, ord=None, axis = axis_rss)