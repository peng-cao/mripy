import numpy as np

"""
back tracking line search from
https://gist.github.com/jiahao/1561144
http://www4.ncsu.edu/~kksivara/ma706/programs/linesearch.m
"""

def BacktrackingLineSearch(f, df, x, p,c = 0.0001,rho = 0.5):
    """
    Backtracking linesearch
    f: function
    x: current point
    p: direction of search
    df: gradient of f at x
    """
    derphi = np.sum(np.multiply(p, df(x)))

    alphak = 1.0
    i = 0

    #Loop
    while i < 20 and f(x + alphak * p) > f(x) + c * alphak * derphi:
        alphak = alphak * rho
        i += 1
    if alphak < 0.1:
        alphak = 0.1
    return alphak, i

"""
back tracking line search from
wiki version
and sparse MRI's implementation
"""

def BacktrackingLineSearch2(f, df, x, p,c = 0.0001,rho = 0.5):
    """
    Backtracking linesearch
    f: function
    x: current point
    p: direction of search
    df: gradient of f at x
    """
    derphi = np.absolute(np.sum(np.multiply(p, df(x))))
    f0 = f(x)
    alphak = 1.0
    i = 0

    #Loop
    while i < 5 and f(x + alphak * p) - f0 >  - c * alphak * derphi:
        alphak = alphak * rho
        i += 1
    #if i > 1:
    #    alphak = alphak * rho
    #if i < 1:
    #    alphak = alphak / rho

    return alphak, i
