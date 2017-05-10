import numpy as np
import pyfftw
#import affinity
import multiprocessing
 
#affinity.set_process_affinity_mask(0,2**multiprocessing.cpu_count()-1)

pyfftw.interfaces.cache.enable()
# fft21e
def fftw1d( data, axes = (0,), threads = 1 ):
    nx = data.shape[0]
    a = pyfftw.empty_aligned(nx, dtype='complex128')
    b = pyfftw.empty_aligned(nx, dtype='complex128')

    fft_object = pyfftw.FFTW(a, b, axes = axes, threads=num_core)
    a[:] = data
    iftdata = fft_object()
    return iftdata

def ifftw1d( data, axes = (0,), threads = 1 ):
    nx = data.shape[0]
    b = pyfftw.empty_aligned(nx, dtype='complex128')
    c = pyfftw.empty_aligned(nx, dtype='complex128')

    ifft_object = pyfftw.FFTW(b, c, axes=axes, direction='FFTW_BACKWARD', threads=threads)
    b[:] = data
    return ifft_object()

def fftw2d( data, axes = (0,1), threads = 1):
    a = pyfftw.empty_aligned(data.shape, dtype='complex128')
    b = pyfftw.empty_aligned(data.shape, dtype='complex128')

    fft_object = pyfftw.FFTW(a, b, axes=axes, threads=threads)
    a[:] = data
    iftdata = fft_object()
    return iftdata

def ifftw2d( data, axes = (0,1), threads = 1 ):
    b = pyfftw.empty_aligned(data.shape, dtype='complex128')
    c = pyfftw.empty_aligned(data.shape, dtype='complex128')

    ifft_object = pyfftw.FFTW(b, c, axes=axes, direction='FFTW_BACKWARD', threads=threads)
    b[:] = data
    return ifft_object()

def fftwnd( data, axes = (0,1,2), threads = 1 ):
    a = pyfftw.empty_aligned(data.shape, dtype='complex128')
    b = pyfftw.empty_aligned(data.shape, dtype='complex128')

    fft_object = pyfftw.FFTW(a, b, axes=axes, threads=threads)
    a[:] = data
    iftdata = fft_object()
    return iftdata

def ifftwnd( data, axes = (0,1,2), threads = 1):
    b = pyfftw.empty_aligned(data.shape, dtype='complex128')
    c = pyfftw.empty_aligned(data.shape, dtype='complex128')

    ifft_object = pyfftw.FFTW(b, c, axes=axes, direction='FFTW_BACKWARD', threads=threads)
    b[:] = data
    return ifft_object()

def test1():
    #print('fftw1d')
    ar, ai  = np.random.randn(2, 8000)
    data    = ar + 1j*ai
    #data    = np.ones(16,np.complex128)
    ftdata  = fftw1d(data,threads=multiprocessing.cpu_count())
    #ftdata2 = np.fft.fft(data)
    iftdata = ifftw1d(ftdata,threads=multiprocessing.cpu_count())
    #print(np.allclose(ftdata, ftdata2))
    print(np.allclose(data, iftdata))


def test2():
    #print('fftw2d')
    #ar, ai  = np.random.randn(2, 8000,8000)
    #data    = ar + 1j*ai
    data    = np.ones((16,16),np.complex128)
    ftdata  = fftw2d(data,threads=multiprocessing.cpu_count())
    #ftdata2 = np.fft.fft2(data)
    iftdata = ifftw2d(ftdata,threads=multiprocessing.cpu_count())

    #print(np.allclose(ftdata, ftdata2))
    print(np.allclose(data, iftdata))

def test3():
    #print('fftw3d')
    N = 128
    data  = np.random.randn(N, N, N, 5) + 1j*np.random.randn(N, N, N, 5)
    #data    = np.ones((16,16,16,2000),np.complex128)
    ftdata  = fftwnd(data,axes=(0,1,2),threads=multiprocessing.cpu_count())
    #ftdata2 = np.fft.fftn(data,axes=(0,1,2))
    iftdata = ifftwnd(ftdata,axes=(0,1,2),threads=multiprocessing.cpu_count())
    
    #print(np.allclose(ftdata, ftdata2))
    print(np.allclose(data, iftdata))