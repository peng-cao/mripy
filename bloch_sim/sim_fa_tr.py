import numpy as np

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
