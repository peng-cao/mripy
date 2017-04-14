"""
test tv denoising 
usage:
execfile('cs_test4.py')
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
import pics.proximal_func as pf
import pics.CS_MRI_solvers_func as solvers
import pics.tvop_func as tv
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

#plotim1(np.absolute(mask))
# define A and invA fuctions, i.e. A(x) = b, invA(b) = x
def Afunc(im):
    ksp = np.fft.fft2(im)
    ksp = np.fft.fftshift(ksp,(0,1))
    return np.multiply(ksp,mask)

def invAfunc(ksp):
    ksp = np.fft.ifftshift(ksp,(0,1))
    im = np.fft.ifft2(ksp)
    return im

def Tfunc(im):
    return tv.D(im)

def Tfunc(dim):
    return tv.adjD(dim)

b = Afunc(x)
#plotim1(np.absolute(b))
#plotim1(np.absolute(invAfunc(b)))

tv_x = tv.grad(x)
#plotim1(np.absolute(tv.grad(x)[:,:,0]))
#plotim1(np.absolute(tv.grad(x)[:,:,1]))

xhat = tv.Div(tv_x)
#plotim1(np.absolute(xhat))

y = x#np.absolute(invAfunc(b))#input image for tv
rho = 0.1#step size
lambda_tv = 10#||f-y||+lambda*TV
#ntheta = 1
sizeg = tv_x.shape #size of gradient tensor
G = np.zeros(sizeg)#intial gradient tensor
i = 0
amp = lambda u : np.sqrt(np.sum(u ** 2,axis=2))

while i < 40:
    dG = tv.grad(tv.Div(G)-y/lambda_tv)
    G = G - rho*dG
    d = np.tile(amp(G)[:,:,np.newaxis], (1,1,2))#.reshape(sizeg)
    G = G/np.maximum(d,1.0*np.ones(sizeg))
    i = i + 1

    #lambda_tv = lambda_tv*ntheta/np.linalg.norm(f-y)
    #print np.linalg.norm(G)

f = y - lambda_tv * tv.Div(G)
plotim1(np.absolute(f))


#do soft thresholding
Nite = 80 #number of iterations
step = 1 #step size
th = 1000 # theshold level
#xopt = solvers.IST_2(Afunc,invAfunc,x,b, Nite, step,th)
#xopt = solvers.ADMM_l2Afxnb_l1x( Afunc, invAfunc, b, Nite, step, 100, 1 )
#xopt = solvers.ADMM_l2Afxnb_l1x_2( Afunc, invAfunc, b, Nite, step, 100, 1 )

#plotim1(np.absolute(xopt))
