import numpy as np
import utilities.utilities_func as ut
"""
function generate fixed fa and tr in arrays 
phase cycling is 0 pi 0 pi
"""
def fatrr(Nk):
    "fixed fa and tr"
    far = np.pi/10*np.ones(Nk)
    far[::2] = -1*far[::2]
    trr = 10*np.ones(Nk)
    return far, trr

def rf_const(Nk, fa):
    "fixed fa, phase 0 pi 0 pi"
    far = fa/180.0 * np.pi * np.ones(Nk)
    far[::2] = -1*far[::2]
    return far

def tr_const(Nk, tr, te = None):
    "fixed tr constant"
    if te is None:
        te = tr/2.0
    trr = tr * np.ones(Nk)
    ter = te * np.ones(Nk)
    return trr, ter

def rftr_const(Nk, fa, tr, te = None):
    far = rf_const(Nk, fa)
    trr,ter = tr_const(Nk, tr, te = te)
    return far, trr, ter

def rf_rand(Nk, fa):
    'generate far with random amplitude and phase'
    far_amp    = np.random.uniform(0, fa/180.0 * np.pi, (Nk,))
    far_phase  = np.random.uniform(-np.pi,         np.pi, (Nk,))
    far        = np.multiply(far_amp, np.exp(far_phase)).astype(np.complex128).squeeze()
    return far


def tr_rand(Nk, trmin, trmax):   
    trr        = np.random.uniform(trmin, trmax, (Nk,)).astype(np.float64).squeeze()	
    return trr

def rftr_rand(Nk, fa, trmin, trmax):
    far = rf_rand(Nk, fa)
    trr = tr_rand(Nk, trmin, trmax)
    return far, trr	

def def_M0():
    return np.array([0.0,0.0,1.0]).astype(np.float64)


def average_dict( dictionary, Ndiv ):
    Ntrperprep        = dictionary.shape[-1]
    Ntrperdiv         = Ntrperprep//Ndiv
    shape_avedict     = list(dictionary.shape)
    shape_avedict[-1] = Ndiv
    shape_averdict    = tuple(shape_avedict)
    avedict           = np.zeros(shape_avedict).astype(np.complex128)
    cnt               = np.zeros(Ndiv)

    for i in range(Ntrperprep):
        idx              = np.minimum(i//Ntrperdiv,Ndiv-1)    
        avedict[...,idx] = avedict[...,idx] + dictionary[...,i]
        cnt[idx]         = cnt[idx] + 1

    for idx in range(Ndiv):
        avedict[...,idx] =  avedict[...,idx]/cnt[idx]
    #ut.plot(cnt)

    return avedict

def average_dict_cnorders( dictionary, cn_orders ):
    Ndiv              = cn_orders.shape[1]
    avedict           = np.dot(dictionary, cn_orders)
    cnt               = np.sum(cn_orders,axis = 0)

    for idx in range(Ndiv):
        avedict[...,idx] =  avedict[...,idx]/cnt[idx]
    #ut.plot(cnt)

    return avedict