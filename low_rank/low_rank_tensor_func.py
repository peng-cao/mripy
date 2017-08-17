"""
should make the changes below in the sktensor code
in pc.py change the type define to complex
sktensor.cp._DEF_TYPE=np.complex
and remove below debug function in line 165 in the pc.py 
log.debug(
'[%3d] fit: %.5f | delta: %7.1e | secs: %.5f' %
(itr, fit, fitchange, exectimes[-1])
)
in lines 154 and 159 added 1e-6

CP decomposition is used

"""

import logging
from scipy.io.matlab import loadmat
from sktensor import dtensor, cp_als
import numpy as np
import utilities.utilities_class as utc
"""
T=1j*np.zeros((8,8))
import numpy as np
#generate a tensor
T=1j*np.zeros((8,8))
T[0,0]=3*1j
# claim a dense tensor
T=dtensor(T)

#P is kurskal operator
P, fit, itr, exectimes = cp_als(T, 3, init='random')
#convert kurskal operator to tensor
#kurskal class is defined in 
#https://github.com/mnick/scikit-tensor/blob/master/sktensor/ktensor.py
T_recon = P.totensor()
"""

def low_rank_tensor_cp(tensor, rank = 2):
    tensor=dtensor(tensor)
    #P is kurskal operator
    P, fit, itr, exectimes = cp_als(tensor, rank, init='random')
    tensor_recon = P.toarray()#P.totensor()
    return tensor_recon

def test():
    N = 256
    #x = 0.001*np.asarray(np.random.rand(N,N,N), np.complex64)
    x = np.zeros((N,N,N),np.complex64)
    x[0:3,0:3] = 10*1j
    timing = utc.timing()
    timing.start()
    x2 = low_rank_tensor_cp(x,3)
    timing.stop().display('low rank tensor ')    
    #print(np.absolute(x).astype(np.float32))
    #print(np.absolute(x2).astype(np.float32)) 
    print(np.allclose(np.absolute(x), np.absolute(x2), atol=1e-2))


if __name__ == "__main__":
    test()
