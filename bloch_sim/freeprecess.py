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

from math import cos, sin, exp
import numpy as np
from numpy.linalg import inv
from numpy.linalg import solve


def freeprecess( T, T1, T2, df ):
   "function simulate free precession and decay"
   phi= 2.*np.pi*df*T/1000.0
   E1 = exp(-T/T1)
   E2 = exp(-T/T2)
   Rz = np.matrix([[cos(phi), -sin(phi), 0.],[sin(phi), cos(phi), 0.],[0., 0., 1.]])

   Afp = np.matrix([[E2, 0., 0.],[0., E2, 0.],[0., 0., E1]])*Rz
   Bfp = np.matrix([0., 0., 1.-E1])
   return Afp, Bfp


