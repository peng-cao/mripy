import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

"""
# color image plot
"""
def plotim1( im ):
    fig, ax = plt.subplots()
    ax.imshow(im)
    ax.axis('off')
    plt.show()
    return

"""
# color image plot, 3d input
"""
def plotim3( im ):
    #im = np.matrix(im)
    nx,ny,nz = im.shape
    #concatenate image along the third dim
    imcat = im[:,:,0]
    for i in range(nz)[1:nz-1]:
        imcat = np.concatenate([imcat, im[:,:,i]],1)

    fig, ax = plt.subplots()
    ax.imshow(imcat)
    ax.axis('off')
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
def plot(x,y=None):
    # Prepare the data
    if y is None:
        y = np.linspace(0, 1, x.size)
    # Plot the data
    plt.plot(x, y)
    # Add a legend
    #plt.legend()
    # Show the plot
    plt.show()

"""
#generate 2d k-space mask
nx: kx dimention size
ny: ky dimention size
center_r: full sampling center area is (2*r)**2
"""

def mask2d( nx, ny, center_r = 15 ):
    #create undersampling mask
    k = int(round(nx*ny*0.5)) #undersampling
    ri = np.random.choice(nx*ny,k,replace=False) #index for undersampling
    ma = np.zeros(nx*ny) #initialize an all zero vector
    ma[ri] = 1 #set sampled data points to 1
    mask = ma.reshape((nx,ny))

    # center k-space index range
    if center_r > 0:
        cx = np.int(nx/2)
        cy = np.int(ny/2)
        cxr = np.arange(round(cx-15),round(cx+15+1))
        cyr = np.arange(round(cy-15),round(cy+15+1))
        mask[np.ix_(map(int,cxr),map(int,cyr))] = np.ones((cxr.shape[0],cyr.shape[0])) #center k-space is fully sampled

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
zero pad the k-space in kx and ky dimentions
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
