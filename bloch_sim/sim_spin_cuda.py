import numpy as np
from numba import cuda
import numba
from math import cos, sin, exp
#from cmath import phase 

# set values for matrix A
@cuda.jit(device=True)
def matgen_cuda( A, e11, e12, e13, e21, e22, e23, e31, e32, e33 ):
    A[0,0] = e11
    A[0,1] = e12
    A[0,2] = e13
    A[1,0] = e21
    A[1,1] = e22
    A[1,2] = e23
    A[2,0] = e31
    A[2,1] = e32
    A[2,2] = e33
    return A

# set values for vector A
@cuda.jit(device=True)
def vecgen_cuda( A, e1, e2, e3 ):
    A[0] = e1
    A[1] = e2
    A[2] = e3
    return A

# B = A , vector copy
@cuda.jit(device=True)
def veccopy_cuda( B, A ):
    B[0] = A[0]
    B[1] = A[1]
    B[2] = A[2]
    return B

# B = A, matrix copy 
@cuda.jit(device=True)
def matcopy_cuda( B, A ):
    for i in range(3):
        for j in range(3):
            B[i, j] = A[i, j]
    return B


#Rz = [[cos(phi), -sin(phi), 0.],[sin(phi), cos(phi), 0.],[0., 0., 1.]]
# rotate around z
@cuda.jit(device=True)
def Rz_cuda( Rz, theta ):
    Rz[0,0] =  cos(theta)
    Rz[0,1] = -sin(theta)
    Rz[0,2] = 0.
    Rz[1,0] =  sin(theta)
    Rz[1,1] =  cos(theta)
    Rz[1,2] = 0.
    Rz[2,0] = 0.
    Rz[2,1] = 0.
    Rz[2,2] = 1.
    return Rz

#Rx = np.matrix([[1., 0., 0.],[0., cos(phi), -sin(phi)],[0., sin(phi), cos(phi)]]) 
#b1 along x, i.e. rotate around x, 
@cuda.jit(device=True)
def Rx_cuda( Rx, phi ):
    Rx[0,0] = 1.
    Rx[0,1] = 0.
    Rx[0,2] = 0.
    Rx[1,0] = 0.
    Rx[1,1] =  cos(phi)
    Rx[1,2] = -sin(phi)
    Rx[2,0] = 0.
    Rx[2,1] =  sin(phi)
    Rx[2,2] =  cos(phi)
    return Rx

#C = A*B
@cuda.jit(device=True)
def matmul_cuda( C, A, B ):
    for i in range(3):
        for j in range(3):
            C[i, j] = 0.
    for i in range(3):
        for j in range(3):
            for k in range(3):
                C[i, j] += A[i, k] * B[k, j]
  
    return C

# c = A * b
@cuda.jit(device=True)
def matmulv_cuda( c, A, b ):
    for i in range(3):
        c[i] = 0.
        for j in range(3):
            c[i] += A[i, j] * b[j]
    return c

#c = a * B
@cuda.jit(device=True)
def vmulmat_cuda( c, a, B ):
    for k in range(3):
        c[k] = 0.
        for j in range(3):
            c[k] += a[j] * B[j,k]
    return c

#C = A.T
@cuda.jit(device=True)
def trans_cuda( C, A ):
    for i in range(3):
        for j in range(3):
            C[i, j] = A[j, i]    
    return C

# A = A * s
@cuda.jit(device=True)
def matmuls_cuda( A, s ):
    for i in range(3):
        for j in range(3):
            A[i, j] = A[i, j]*s    
    return A

# a = a*s
@cuda.jit(device=True)
def vmuls_cuda( a, s ):
    for i in range(3):
        a[i] = a[i]*s    
    return a

# a += b
@cuda.jit(device=True)
def vaddv_cuda( a, b ):
    for i in range(3):
        a[i] = a[i]+b[i]    
    return a


# a += b
@cuda.jit(device=True)
def xynull_cuda( a ):
    for i in range(2):
        a[i] = 0    
    return a
"""
gpu version, function rotating spin in SO(3), according to Euler angular
This function need these input tmp arrays
in kernel function, one shold claim local memory for them
Rz  = cuda.local.array(shape=(3, 3), dtype=numba.float64)
Rx  = cuda.local.array(shape=(3, 3), dtype=numba.float64)
Rth = cuda.local.array(shape=(3, 3), dtype=numba.float64)

Output array
in kernel function, one shold claim local memory for them 
Rtho = cuda.local.array(shape=(3, 3), dtype=numba.float64)
"""
@cuda.jit(device=True)
def throt_cuda( Rtho, Rz, Rx, Rth, phi, theta ):
    #set Rx to phi
    Rx_cuda(Rx, phi)
    #set Rz to theta 
    Rz_cuda(Rz, theta)
    matmul_cuda(Rth,Rz,Rx)
    #set Rz to -theta
    Rz_cuda(Rz, -theta)   
    matmul_cuda(Rtho,Rth,Rz) #not inv(Rz)*Rx*Rz,
    return Rtho

"""
gpu version Function simulates free precession and decay
over a time interval T, given relaxation times T1 and T2
and off-resonance df.  Times in ms, off-resonance in Hz.

this function need input of these tmp arrays
in kernel function, one shold claim local memory for them
Rz  = cuda.local.array(shape=(3, 3), dtype=numba.float32)
Em  = cuda.local.array(shape=(3, 3), dtype=numba.float32)

oupt arrays 
in kernel function, one shold claim local memory for them
Afp = cuda.local.array(shape=(3, 3), dtype=numba.float32)
Bfp = cuda.local.array(shape=3,      dtype=numba.float32)
"""
@cuda.jit(device=True)
def cal_freeprecess_cuda( Afp, Bfp, Rz, Em, T, T1, T2, df ):
    "function simulate free precession and decay"
    phi= 2.*np.pi*df*T/1000.0 #offset freq relaxation
    E1 = exp(-T/T1) #T1 relaxation 
    E2 = exp(-T/T2) #T2 relzxation
    # set Rz to phi
    Rz = Rz_cuda(Rz, phi)
    # set T1 and T2 relaxation
    matgen_cuda(Em, E2, 0., 0.,0., E2, 0.,0., 0., E1)
    # Afp = Em * Rz
    matmul_cuda(Afp, Em, Rz)
    # Bfp = 1-E1, T1 recovery term
    vecgen_cuda(Bfp, 0., 0., 1.-E1)
    # output Afp and Bfp in this function
    return Afp, Bfp

@cuda.jit(device=True)
def excitation_cuda( Mtmp, M, Rtho, Rz, Rx, Rth, phi, theta ):
    # calculate Rtho rotation matrix, for excitation 
    # it takes two parameters: phi (rf amplitude) and theta (rf phase)
    throt_cuda( Rtho, Rz, Rx, Rth, phi, theta )
    # apply Rtho, Mtmp = Rtho * M
    matmulv_cuda(Mtmp,Rtho,M)
    return Mtmp

@cuda.jit(device=True)
def relaxation_cuda( Mtmp, M, Afp, Bfp, Rz, Em, T, T1, T2, df, PD ):
    # pre-calculate Afp and Bfp from parameters
    cal_freeprecess_cuda( Afp, Bfp, Rz, Em, T, T1, T2, df )
    # apply Afp, Mtmp = Afp * M
    matmulv_cuda(Mtmp, Afp, M)
    # apply Bfp, first Bfp is PD weighted
    # Mtmp = Mtmp + Bfp*PD
    vmuls_cuda(Bfp, PD)
    vaddv_cuda(Mtmp, Bfp)
    return Mtmp

@cuda.jit
def test_cuda(M0):
    i  = cuda.grid(1)
    if i > 30:
        return
    
    phi= -0.5 * np.pi # rf amplitude
    theta= 0. * np.pi # rf phase
    T = 0.1 # relaxation time
    T1 = 1. # T1
    T2 = 1000. # T2
    df = 0.  # freq offset
    PD = 0.5  # proton density

    # claim local memory
    Rz   = cuda.local.array(shape=(3, 3), dtype=numba.float64)
    Rx   = cuda.local.array(shape=(3, 3), dtype=numba.float64)
    Mtmp = cuda.local.array(shape=3,      dtype=numba.float64)
    M    = cuda.local.array(shape=3,      dtype=numba.float64)    
    Rth  = cuda.local.array(shape=(3, 3), dtype=numba.float64)
    Rtho = cuda.local.array(shape=(3, 3), dtype=numba.float64)
    Em   = cuda.local.array(shape=(3, 3), dtype=numba.float64)#float32
    Afp  = cuda.local.array(shape=(3, 3), dtype=numba.float64)#float32
    Bfp  = cuda.local.array(shape=3,      dtype=numba.float64)#float32
 

    #simple test    
    #Rz_cuda(Rz, phi)
    #Rx_cuda(Rx, theta)
    #matmulv_cuda(Mtmp,Rz,M)
    #veccopy_cuda(M, Mtmp)

    # M0=[0 0 1] should be proton density weighted
    veccopy_cuda(M, M0)
    vmuls_cuda(M, PD)
    #excitation
    throt_cuda( Rtho, Rz, Rx, Rth, phi, theta )
    matmulv_cuda(Mtmp,Rtho,M)
    veccopy_cuda(M, Mtmp)
    #relaxation
    cal_freeprecess_cuda( Afp, Bfp, Rz, Em, T, T1, T2, df )
    matmulv_cuda(Mtmp, Afp, M)
    vmuls_cuda(Bfp, PD)
    vaddv_cuda(Mtmp, Bfp)
    veccopy_cuda(M, Mtmp)
    #output
    veccopy_cuda(M0, M)       

if __name__ == "__main__":
    n = 100#x.shape[0] #number of kernels in the computing
    M  = np.array([0., 0., 1.])
    device = cuda.get_current_device()
    tpb = device.WARP_SIZE
    bpg = int(np.ceil(float(n)/tpb))
    test_cuda[bpg, tpb](M)
    print(M)
