"""
rotating spin
"""
import numpy as np
from math import cos, sin
from numpy.linalg import inv
from numpy.linalg import solve

def throt( phi, theta ):
   "rotate spin"
   Rz = np.matrix([[cos(-theta), -sin(-theta), 0],[sin(-theta), cos(-theta), 0],[0, 0, 1]])
   Rx = np.matrix([[1, 0, 0],[0, cos(phi), -sin(phi)],[0, sin(phi), cos(phi)]])
   #invRz = inv(Rz)
   Rth = solve(Rz,Rx)*Rz #inv(Rz)\Rx*Rz, which is persudo_inv(inv(Rz))*Rx*Rz
   return Rth
