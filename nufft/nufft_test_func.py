from __future__ import print_function, division
import numpy as np
import numba
from numba import cuda
from time import time
from math import exp
import utilities.utilities_func as ut
import matplotlib.pyplot as plt

"""
this is a test code from
https://jakevdp.github.io/blog/2015/02/24/optimizing-python-with-numpy-and-numba/
it compare the input nufft_func with the nudft1d1
"""
def compare_nufft1d1( nufft_func1, nufft_func2, ms=1000, mc=100000 ):
    # Test vs the direct method
    print(30 * '-')
    rng = np.random.RandomState(0)
    x = 100 * rng.rand(mc)
    c = 1j*np.sin(x) + 1j*np.sin(10*x)
    for df in [1, 2.0]:
        for iflag in [1, -1]:
            print ("testing 1d df=%f, iflag=%f"% (df,iflag))
            F1 = nufft_func1(x, c, ms, df=df, iflag=iflag)
            F2 = nufft_func2(x, c, ms, df=df, iflag=iflag)
            ut.plot(np.absolute(F1))
            ut.plot(np.absolute(F2))
            assert np.allclose(F1, F2, rtol=1e-02, atol=1e-02)
    print("- Results match the 1d DFT")

def compare_nufft2d1( nufft_func1, nufft_func2, ms=1000, mt=1000, mc=100000 ):
    # Test vs the direct method
    print(30 * '-')
    rng = np.random.RandomState(0)
    x = 100 * rng.rand(mc)
    y = 100 * rng.rand(mc)
    c = np.sin(2*x) + 1j*np.cos(2*y)#np.ones(x.shape)#
    for df in [1.0, 2.0]:
        for iflag in [1, -1]:
            print ("testing 2d df=%f, iflag=%f"% (df,iflag))
            F1 = nufft_func1(x, y, c, ms, mt, df=df, iflag=iflag)
            F2 = nufft_func2(x, y, c, ms, mt, df=df, iflag=iflag)
            ut.plotim1(np.absolute(F1), colormap = None, title = None, bar = True)
            ut.plotim1(np.absolute(F2), colormap = None, title = None, bar = True)
            ut.plotim1(np.absolute(F1-F2), colormap = None, title = None, bar = True)
            assert np.allclose(F1, F2, rtol=1e-02, atol=1e-02)
    print("- Results match the 2d DFT")

def compare_nufft3d1( nufft_func1, nufft_func2, ms=1000, mt=1000, mu=1000, mc=100000 ):
    # Test vs the direct method
    print(30 * '-')
    rng = np.random.RandomState(0)
    x = 100 * rng.rand(mc)
    y = 100 * rng.rand(mc)
    z = 100 * rng.rand(mc)
    c = np.sin(2*x) + 1j*np.cos(2*y)
    for df in [1]:
        for iflag in [1, -1]:
            print ("testing 3d df=%f, iflag=%f"% (df,iflag))
            F1 = nufft_func1(x, y, z, c, ms, mt, mu, df=df, iflag=iflag)
            F2 = nufft_func2(x, y, z, c, ms, mt, mu, df=df, iflag=iflag)
            ut.plotim1(np.absolute(F1[:,:,mu//2]), colormap = None, title = None, bar = True)
            ut.plotim1(np.absolute(F2[:,:,mu//2]), colormap = None, title = None, bar = True)
            ut.plotim1(np.absolute(F1[:,:,mu//2]-F2[:,:,mu//2]), colormap = None, title = None, bar = True)
            assert np.allclose(F1, F2, rtol=1e-02, atol=1e-02)
    print("- Results match the 3d DFT")


def compare_nufft1d21( nufft_func1, nufft_func2, ms=1000, mc=100000, Reptime=1 ):
    rng = np.random.RandomState(0)
    # Time the nufft function
    x = 100 * rng.rand(mc)
    c0 = np.sin(3*x) + 0.1*1j*np.sin(4.5*x)
    F1 = nufft_func1(x, c0, ms)
    times = []
    for i in range(Reptime):
        t0 = time()
        F2 = nufft_func2(x, F1, ms)
        t1 = time()
        times.append(t1 - t0)
        ut.plot(np.absolute(F1))
        ut.plot(np.absolute(F2))
        #ut.plot(np.real(F1),np.real(F2),'o')
    print("- Execution time (M={0}): {1:.2g} sec".format(mc, np.median(times)))

# timing for the input nufft_func, the nufft_func should be 1d nufft function type 1
def time_nufft1d1( nufft_func, ms=1000, mc=100000, Reptime=5 ):
    rng = np.random.RandomState(0)
    # Time the nufft function
    x = 100 * rng.rand(mc)
    c = np.sin(2*x)
    times = []
    for i in range(Reptime):
        t0 = time()
        F = nufft_func(x, c, ms)
        t1 = time()
        times.append(t1 - t0)
    print("- Execution time (M={0}): {1:.2g} sec".format(mc, np.median(times)))

# timing for the input nufft_func, the nufft_func should be 1d nufft function type 2
def time_nufft1d2( nufft_func1, nufft_func2, ms=1000, mc=100000, Reptime=5 ):
    rng = np.random.RandomState(0)
    # Time the nufft function
    x = 100 * rng.rand(mc)
    c0 = np.sin(x)
    F = nufft_func1(x, c0, ms)
    times = []
    for i in range(Reptime):
        t0 = time()
        c = nufft_func2(x, F, ms)
        t1 = time()
        times.append(t1 - t0)
    ut.plot(x,np.real(c0),'o')
    ut.plot(x,np.real(c),'o')
    ut.plot(np.real(c0),np.real(c),'o')
    print("- Execution time (M={0}): {1:.2g} sec".format(mc, np.median(times)))

# time for nufft_func, the nufft_func should be a 2d nufft function type 1
def time_nufft2d1( nufft_func, ms=1000, mt=1000, mc=1000, Reptime=5 ):
    rng = np.random.RandomState(0)
    # Time the nufft function
    x = 100 * rng.rand(mc)
    y = 100 * rng.rand(mc)
    #c = np.sin(x) #two peaks in transformed space along x
    c = np.sin(y) #two peaks in transformed space along y
    #c = np.ones(x.shape) #one peak in the center of transformed space
    times = []
    for i in range(Reptime):
        t0 = time()
        F = nufft_func(x, y, c, ms, mt)
        t1 = time()
        times.append(t1 - t0)
        ut.plotim1(np.absolute(F), colormap = None, title = None, bar = True)
    print("- Execution time (M={0}): {1:.2g} sec".format(mc, np.median(times)))

# timing for the input nufft_func, the nufft_func should be 2d nufft function type 2
def time_nufft2d2( nufft_func1, nufft_func2, ms=1000, mt=1000, mc=100000, Reptime=5 ):
    rng = np.random.RandomState(0)
    # Time the nufft function
    x = 100 * rng.rand(mc)
    y = 100 * rng.rand(mc)
    c0 = np.sin(x)
    F = nufft_func1(x, y, c0, ms, mt)
    times = []
    for i in range(Reptime):
        t0 = time()
        c = nufft_func2(x, y, F, ms, mt)
        t1 = time()
        times.append(t1 - t0)
    ut.plot(x,np.real(c0),'o')
    ut.plot(x,np.real(c),'o')
    ut.plot(np.real(c0),np.real(c),'o')
    print("- Execution time (M={0}): {1:.2g} sec".format(mc, np.median(times)))

# time for nufft_func, the nufft_func should be a 3d nufft function type 1
def time_nufft3d1( nufft_func, ms=1000, mt=1000, mu=1000, mc=1000, Reptime=5 ):
    rng = np.random.RandomState(0)
    # Time the nufft function
    x = 100 * rng.rand(mc)
    y = 100 * rng.rand(mc)
    z = 100 * rng.rand(mc)
    #c = np.sin(x) #two peaks in transformed space along x
    c = np.sin(y) #two peaks in transformed space along y
    #c = np.ones(x.shape) #one peak in the center of transformed space
    times = []
    for i in range(Reptime):
        t0 = time()
        F = nufft_func(x, y, z, c, ms, mt, mu)
        t1 = time()
        times.append(t1 - t0)
        ut.plotim1(np.absolute(F[:,:,mu//2]), colormap = None, title = None, bar = True)
    print("- Execution time (M={0}): {1:.2g} sec".format(mc, np.median(times)))

# timing for the input nufft_func, the nufft_func should be 3d nufft function type 2
def time_nufft3d2( nufft_func1, nufft_func2, ms=1000, mt=1000, mu=1000, mc=100000, Reptime=5 ):
    rng = np.random.RandomState(0)
    # Time the nufft function
    x = 100 * rng.rand(mc)
    y = 100 * rng.rand(mc)
    z = 100 * rng.rand(mc)
    c0 = np.sin(x)
    F = nufft_func1(x, y, z, c0, ms, mt, mu)
    times = []
    for i in range(Reptime):
        t0 = time()
        c = nufft_func2(x, y, z, F, ms, mt, mu)
        t1 = time()
        times.append(t1 - t0)
    ut.plot(x,np.real(c0),'o')
    ut.plot(x,np.real(c),'o')
    ut.plot(np.real(c0),np.real(c),'o')
    print("- Execution time (M={0}): {1:.2g} sec".format(mc, np.median(times)))
