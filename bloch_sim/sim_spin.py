
from math import cos, sin, exp
import numpy as np
from numpy.linalg import inv
from numpy.linalg import solve

"""
Function simulates free precession and decay
over a time interval T, given relaxation times T1 and T2
and off-resonance df.  Times in ms, off-resonance in Hz.
useage e.g.
import sim_spin as ss
import numpy as np

M=np.matrix([0,0,1])
A,B=ss.freeprecess(5.,1000.,200.,0.) # TR 5ms, T1 1s, T2 0.2s, df 0Hz
"""
def freeprecess( T, T1, T2, df ):
   "function simulate free precession and decay"
   phi= 2.*np.pi*df*T/1000.0
   E1 = exp(-T/T1)
   E2 = exp(-T/T2)
   Rz = np.matrix([[cos(phi), -sin(phi), 0.],[sin(phi), cos(phi), 0.],[0., 0., 1.]])

   Afp = np.matrix([[E2, 0., 0.],[0., E2, 0.],[0., 0., E1]])*Rz
   Bfp = np.matrix([0., 0., 1.-E1])
   return Afp, Bfp

"""
Function wrap freeprecess , not test yet
"""
def freeprecesswrap( M0, M, T, T1, T2, df ):
   "function simulate free precession"
   A,B = ss.freeprecess(T,T1,T2,df)
   M = A*M+M0*B.T
   return M

"""
function rotating spin in SO(3), according to Euler angular
usage e.g.
import sim_spin as ss
import numpy as np

M=np.matrix([0,0,1])
ss.throt(pi/2,0)*M.T #90deg flip
ss.throt(pi/2,pi/3+pi)*ss.throt(pi/2,pi/3)*M.T  #60deg tip down and 60deg tip up
"""
def throt( phi, theta ):
   "rotate spin"
   Rz = np.matrix([[cos(-theta), -sin(-theta), 0.],[sin(-theta), cos(-theta), 0.],[0., 0., 1.]])#b1 along z axis,i.e. rotate around z
   Rx = np.matrix([[1., 0., 0.],[0., cos(phi), -sin(phi)],[0., sin(phi), cos(phi)]]) #b1 along x, i.e. rotate around x
   #invRz = inv(Rz)
   Rth = solve(Rz,Rx)*Rz #not inv(Rz)\Rx*Rz, which is persudo_inv(inv(Rz))*Rx*Rz
   return Rth
