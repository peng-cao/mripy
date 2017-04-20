import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.io as sio
"""
# color image plot
"""
def plotim1( im, colormap = None, title = None, bar = None ):
    fig, ax = plt.subplots()
    if colormap is None:
        cax = ax.imshow(im, cmap = cm.gray, origin='lower', interpolation='none')
    else:
        cax = ax.imshow(im, cmap = colormap, origin='lower', interpolation='none')
    ax.axis('off')
    if title is not None:
        ax.set_title(title)
    if bar is not None:
        cbar = fig.colorbar(cax)
        #cbar.ax.set_yticklabels([str(bar_ticks[0]), str(bar_ticks[-1:])]) 
    plt.show()
    return


"""
# color image plot, 3d input
# concatenate image along the third dim
"""
def plotim3( im, catdim = [10,-1] , colormap = None, title = None, bar = None ):
    #im = np.matrix(im)
    nx,ny,nz = im.shape
    #concatenate image in 1d, along the third dim

    if catdim[0] >= nz :
        imcat = im[:,:,0]        
        for i in range(nz)[1:nz-1]:
            imcat = np.concatenate([imcat, im[:,:,i]],1)
    else:
        # compute concatenate dim
        nz_catx = min(nz, catdim[0])
        if catdim[1] is not -1:
            nz_caty = min(nz//nz_catx, catdim[1])
        else:
            nz_caty = nz//nz_catx
        #intial the cat image
        imcatx = np.zeros((nx*nz_catx, ny))        
        # zero pad im if nz < nz_catx * nz_caty
        if nz < nz_catx * nz_caty:
            im = np.concatenate([im, np.zeros((nx,ny,nz_catx*nzcaty-nz))], 2)
        #concatenate image in 2d, along x and y
        for j in range(nz_caty):
            imcatx = im[:,:,j*nz_catx]
            for i in range(nz_catx):
                imcatx = np.concatenate([imcatx, im[:,:,i + j*nz_catx]],1)#concatenate along x
            if j is 0:
                imcat = imcatx
            else:
                imcat = np.concatenate([imcat,imcatx],0)#concatenate along y
    fig, ax = plt.subplots()
    #ax.imshow(imcat)
    if colormap is None:
        cax = ax.imshow(imcat, cmap = cm.gray, origin='lower', interpolation='none')
    else:
        cax = ax.imshow(imcat, cmap = colormap, origin='lower', interpolation='none')
    ax.axis('off')
    if title is not None:
        ax.set_title(title)
    if bar is not None:
        cbar = fig.colorbar(cax)
    plt.show()
    return

"""
#gray image plot
"""
def plotgray( im ):
    fig, ax = plt.subplots()
    ax.imshow(im, cmap=cm.gray)
    ax.axis('off')
    plt.show()
    return

"""
plot a line
"""
def plot( x, y=None, line_type = '-', legend = None ):
    # Prepare the data
    if y is None:
        y = np.linspace(0, 1, x.size)
        plt.plot(y, x, line_type)
    else:
        plt.plot(x, y, line_type)
    # Add a legend
    #plt.legend()
    if legend is not None:
        plt.legend(legend)    
    # Show the plot
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
        cxr = np.arange(round(cx-15),round(cx+15))
        cyr = np.arange(round(cy-15),round(cy+15))
        
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
        cxr = np.arange(round(cx-15),round(cx+15))
        cyr = np.arange(round(cy-15),round(cy+15))
        czr = np.arange(round(cz-15),round(cz+15))
        
    return data[np.ix_(map(int,cxr),map(int,cyr),map(int,czr))]

"""
zero pad the 3d k-space in kx and ky dimentions
"""
def pad3d( data, nx, ny, nz ):
    #create undersampling mask
    datsize = data.shape
    ndata = 1j*np.zeros((nx,ny,nz,datsize[3],datsize[4]))

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
zero pad the 2d k-space in kx and ky dimentions
""" 

def pad2d( data, nx, ny ):
    #create undersampling mask
    datsize = data.shape
    ndata = 1j*np.zeros((nx,ny,datsize[2],datsize[3]))

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

