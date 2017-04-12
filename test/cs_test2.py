"""
test soft thresholding algrithom
usage:
execfile('cs_test2.py')
"""


# make sure you've got the following packages installed
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize as spopt
import scipy.fftpack as spfft
import scipy.ndimage as spimg
import scipy.io as sio
import importlib
import matplotlib.cm as cm
import proximal_func as pf
import IST

def plotim1(im):
    fig, ax = plt.subplots()
    ax.imshow(im)
    ax.axis('off')
    plt.show()

def plotim2(im):
    fig, ax = plt.subplots()
    ax.imshow(im, cmap=cm.gray)
    ax.axis('off')
    plt.show()

# simulated image
mat_contents = sio.loadmat('sim_2dmri.mat');
x = mat_contents["sim_2dmri"]
#x = spimg.zoom(xorig, 0.4)
#plotim2(x)

nx,ny = x.shape

#create undersampling mask
k = int(round(nx*ny*0.5)) #undersampling
ri = np.random.choice(nx*ny,k,replace=False) #index for undersampling
ma = np.zeros(nx*ny) #initialize an all zero vector
ma[ri] = 1 #set sampled data points to 1
mask = ma.reshape((nx,ny))

cx = np.int(nx/2)
cy = np.int(ny/2)
cxr = np.arange(round(cx-15),round(cx+15+1))
cyr = np.arange(round(cy-15),round(cy+15+1))

mask[np.ix_(map(int,cxr),map(int,cyr))] = np.ones((cxr.shape[0],cyr.shape[0])) #center k-space is fully sampled

plotim1(np.absolute(mask))

# define A and invA fuctions, i.e. A(x) = b, invA(b) = x
def Afunc(im):
    ksp = np.fft.fft2(im)
    ksp = np.fft.fftshift(ksp,(0,1))
    return np.multiply(ksp,mask)

def invAfunc(ksp):
    ksp = np.multiply(ksp,mask)
    ksp = np.fft.ifftshift(ksp,(0,1))
    im = np.fft.ifft2(ksp)
    return im

b = Afunc(x)
plotim1(np.absolute(b))
plotim1(np.absolute(invAfunc(b)))

#do soft thresholding
Nite = 150 #number of iterations
step = 0.5 #step size
th = 1000 # theshold level
xopt = IST.IST_2(Afunc,invAfunc,b, Nite, step,th)

plotim1(np.absolute(xopt))
