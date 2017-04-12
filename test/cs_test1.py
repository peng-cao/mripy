"""
test soft thresholding algrithom
usage:
execfile('cs_test1.py')
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
xorig = mat_contents["sim_2dmri"]
x = spimg.zoom(xorig, 0.4)
#plotim2(x)

nx,ny = x.shape

#create undersampling mask
k = int(round(nx*ny*0.5)) #undersampling
ri = np.random.choice(nx*ny,k,replace=False) #index for undersampling
ma = np.zeros(nx*ny) #initialize an all zero vector
ma[ri] = 1 #set sampled data points to 1
maskdiag = np.diag(ma) #mask is a diagonal matrix
np.delete(maskdiag,ri,0) #remove the all-zero rows in mask matrix

#2d dct = kron(1ddct,1ddct)
aa = spfft.dct( np.identity(nx),norm='ortho',axis=0)
bb = spfft.dct(np.identity(ny),norm='ortho',axis=0)
A = np.kron( aa , bb  )#2d dct

#apply mask to FT operator, Ax = b
A = maskdiag*A
b = A.dot(x.flatten())

# define A and invA fuctions, i.e. A(x) = b, invA(b) = x

#do soft thresholding
Nite = 50 #number of iterations
step = 0.1 #step size
th = 0.1 # theshold level
Xopt = IST.IST_1(A,b, Nite, step,th)

plotim1(np.absolute(Xopt.reshape((nx,ny))))
