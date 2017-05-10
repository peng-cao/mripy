import numpy as np
from opt_alg import BacktrackingLineSearch2
import tvop_func as tv
import tvop_class as tv_class
"""
softthreshold/proximal for l1 norm, th = lambda/rho
argmin_x (lambda)*||x||_1 + (rho/2)*||x-x0||_2^2

"""
# output always complex data type
def prox_l1_soft_thresh( x0, th ):
    a_th = np.abs(x0) - th
    a_th[a_th<0] = 0
    a_angle = np.angle(x0)
    return np.multiply(np.exp(1j*a_angle), a_th)

#modified, input float type, output float type
def prox_l1_soft_thresh2( x0, th ):
    a_th = np.abs(x0) - th
    a_th[a_th<0] = 0
    a_dir = np.divide(x0,np.abs(x0)+1e-6)
    return np.multiply(a_dir, a_th)

# hard threshold
def prox_l0_hard_thresh( x0, th ):
    a_th = np.abs(x0) #- th
    a_th[a_th<th] = 0
    a_dir = np.divide(x0,np.abs(x0)+1e-6)
    return np.multiply(a_dir, a_th)

"""
softthreshold for proximal transformed l1 norm, th = lambda/rho
argmin_x (lambda)*||Tfunc(x)||_1 + (rho/2)*||x-x0||_2^2 
"""
# output is always complex type
def prox_l1_Tf_soft_thresh( Tfunc, invTfunc, x0, th ):
    Tfx0 = Tfunc(x0)
    a_th = np.abs(Tfx0) - th
    a_th[a_th<0] = 0
    a_angle = np.angle(Tfx0)
    Tfx = np.multiply(np.exp(1j*a_angle), a_th)
    return invTfunc(Tfx)
#input float type, output float type
def prox_l1_Tf_soft_thresh2( Tfunc, invTfunc, x0, th ):
    Tfx0 = Tfunc(x0)
    a_th = np.abs(Tfx0) - th
    a_th[a_th<0] = 0
    a_dir = np.divide(Tfx0,np.abs(Tfx0)+1e-6)
    Tfx = np.multiply(a_dir, a_th)
    return invTfunc(Tfx)

"""
total variation minimization
2d input
argmin_f ||f-y||_2^2 + lambda*J(y), where J(y) is ||y||_TV

the proximal function shold have the form of
argmin_f ||y||_TV + (rho/2)*||f-y||_2^2

so the lambda shall be set as 2*tv_r/rho, tv_r is regularization parameter

method from
Chambolle, An algorithm for total variation minimizations and applications, 2004
and a pdf file
Total Variation Regularization with Chambolle Algorihtm.pdf
"""
#for 2d input data
def prox_tv2d( y, lambda_tv, step = 0.1 ):
    #lambda_tv = 2/rho
    nx, ny = y.shape
    sizeg = (nx,ny,2) #size of gradient tensor
    G = np.zeros(sizeg)#intial gradient tensor
    i = 0

    amp = lambda u : np.sqrt(np.sum(u ** 2,axis=2))#nomalize u along the third dimension

    while i < 40:
        dG = tv.grad(tv.Div(G)-y/lambda_tv)#gradient of G
        G = G - step*dG#gradient desent, tested to work with negative sign for gradient update
        d = np.tile(amp(G)[:,:,np.newaxis], (1,1,2))#.reshape(sizeg)
        G = G/np.maximum(d,1.0*np.ones(sizeg))#normalize to ensure the |G|<1
        i = i + 1
        #lambda_tv = lambda_tv*ntheta/np.linalg.norm(f-y)
        #print np.linalg.norm(G)
    f = y - lambda_tv * tv.Div(G)

    return f

#for 3d input data
def prox_tv3d( y, lambda_tv, step = 0.1 ):
    #lambda_tv = 2/rho
    #nx, ny, nz = y.shape
    sizeg = y.shape+(y.ndim,) #size of gradient tensor
    G = np.zeros(sizeg)#intial gradient tensor
    i = 0
    tvopt = tv_class.TV3d()
    #amp = lambda u : np.sqrt(np.sum(u ** 2,axis=3))#nomalize u along the third dimension
    #norm_g0 = np.linalg.norm(tvopt.grad(y))
    #norm_g = norm_g0
    while i < 40:
        dG = tvopt.grad(tvopt.Div(G)-y/lambda_tv)#gradient of G
        G = G - step*dG#gradient desent, tested to work with negative sign for gradient update
        d = tvopt.amp(G)#np.tile(amp(G)[:,:,np.newaxis], (1,1,1,2))#.reshape(sizeg)
        G = G/np.maximum(d,1.0*np.ones(sizeg))#normalize to ensure the |G|<1
        i = i + 1
        #lambda_tv = lambda_tv*ntheta/np.linalg.norm(f-y)
        #norm_g = np.linalg.norm(G)
    f = y - lambda_tv * tvopt.Div(G)
    return f

"""
project on the set C, from set C find a vector that is closest to input x0
C can be a dictionary
argmin_x I_C + ||x-x0||_2^2
def prox_pos_dictmatch(C,x0)
"""

"""
project on neuron network, output of neuron network is a vector x that is the closest to input x0 and predicting parmeters y
argmin_x -[log(likelihood(y|x))] + ||x-x0||_2^2
minimizing negtive log-likelihood while ensure x is close to x0 in l2-norm sense
use averaged projection method or alternating projection method
def prox_pos_ML(forward_MLpredict,backward_Blochsim,x0)
"""

"""
Tikhonov regularization/proximal function for l2 norm of Ax-b
argmin_x ||Ax-b||_2^2 + (rho/2)*||x-x0||_2^2
A has to be a 2D matrix and b is a 1d vector
A matrix normal contain Fourier transform and undersampling mask in k-space
A.T is the transpose operator of A

"""
def prox_l2_Axnb( A, b, x0, rho ):
    na, nb = A.shape
    x = np.linalg.inv(A.T.dot(A) + rho*np.identity(na)).dot(A.T.dot(b)+rho*x0)
    return x

"""
pre-compute to save some multiplications for l2 norm of Ax-b
argmin_x [ ||Ax-b||_2^2+(rho/2)*||x-x0||_2^2 ]

"""
def prox_l2_Axnb_precomputpart( A, b, rho ):
    na,nb = A.shape
    A_T_A = A.T.dot(A)
    A_T_b = A.T.dot(b)
    Q = A_T_A + rho*np.identity(na)
    Q = np.lingalg.inv(Q)
    Q_dot = Q.dot
    return Q_dot, A_T_b

"""
compute proximal function for l2 norm of Ax-b, using multiplicatoins from pre_comput_prox_l2_axnb
argmin_x ||Ax-b||_2^2+(rho/2)*||x-x0||_2^2
"""
def prox_l2_Axnb_iterpart( Q_dot, A_T_b, x0, rho ):
     x = Q_dot(A_T_b + rho*x0)
     return x

"""
gradient of f(x) = ||Afunc(x)-b||_2^2+(rho/2)*||x-x0||_2^2
"""
def grad_prox_l2_Afxnb(Afunc,invAfunc, b,x, x0,rho):
    df = 2*invAfunc(Afnc(x)-b)+rho*(x-x0)
    return df

"""
gadient desent for minimizing f(x) = ||Afunc(x)-b||_2^2+(rho/2)*||x-x0||_2^2
"""
def prox_l2_Afxnb_GD( Afunc, invAfunc, b, x0, rho, Nite, step ):
    x = np.zeros(x0.shape)
    eps = 0.001
    r = -2*invAfunc(-b)+rho*x0#zero as intial guess #-2*invAfunc(Afunc(x0)-b)#x=x0 as intial guess, i.e. here r=df(x0)
    delta = np.linalg.norm(r)
    delta0 = delta
    i = 0
    # iteration
    while i < Nite and delta > eps*delta0:
        r = -2*invAfunc(Afunc(x)-b)-rho*(x-x0)#-grad_prox_l2_Afxnb(Afunc,invAfunc, b,x, x0,rho)
        x = x + step*r
        delta = np.linalg.norm(r)
        #print delta
        i = i + 1
    return x

"""
conjugate gradient desent for minimizing f(x) = ||Afunc(x)-b||_2^2+(rho/2)*||x-x0||_2^2
df = 2*invAfunc(Afunc(x)-b) + rho*(x-x0) = 2*invAfunc(Afunc(x)) - 2*invAfunc(b) + rho*x - rho*x0 = Anfunc(x) - bn
where
Anfunc(x) = 2*invAfunc(Afunc(x)) + rho*x
bn = 2*invAfunc(b) + rho*x0
should use nonlinear conjugate gradient method
"""
def prox_l2_Afxnb_CGD( Afunc, invAfunc, b, x0, rho, Nite ):
    #x = np.zeros(x0.shape)
    eps = 0.001
    i = 0
    dx =-2.0*invAfunc(Afunc(x0)-b)#initial is x0# -2.0*invAfunc(-b) + rho*x0 #intial is zero #
    def f(xi):
        return np.linalg.norm(Afunc(xi)-b)+(rho/2)*np.linalg.norm(xi-x0)

    def df(xi):
        return 2*invAfunc(Afunc(xi)-b)+rho*(xi-x0)

    #alpha linear search argmin_alpha f(x0 + alpha*dx)
    alpha,nstp = BacktrackingLineSearch2(f, df, x0, dx)
    x = x0 + alpha * dx
    s = dx
    delta0 = np.linalg.norm(dx)
    deltanew = delta0
    # iteration
    while i < Nite and deltanew > eps*delta0 and nstp < 20:
        dx = -2*invAfunc(Afunc(x)-b)-rho*(x-x0)
        #Fletcher-Reeves: beta = np.linalg.norm(dx)/np.linalg.norm(dx_old)
        deltaold = deltanew
        deltanew = np.linalg.norm(dx)
        beta = float(deltanew / float(deltaold))
        s = dx + beta * s
        #alpha linear search argmin_alpha f(x + alpha*s)
        alpha,nstp = BacktrackingLineSearch2(f, df, x, s)
        x = x + alpha * s
        i = i + 1
        #print nstp
    return x
"""
def prox_l2_Afxnb_CGD2( Afunc, invAfunc, b, x0, rho, Nite ):
    x = np.zeros(x0.shape)
    eps = 0.001
    delta = np.linalg.norm(2*invAfunc(Afunc(x0)-b))
    i = 0

    def Anfunc(xi):
        An = 2*invAfunc(Afunc(xi)) + rho*xi
        return An

    bn = 2*invAfunc(b) + rho*x0
    r = bn - Anfunc(x0) #intial guess of x is x0
    d = r
    deltanew = np.linalg.norm(r)
    delta0 = deltanew
    # iteration
    while i < Nite and delta > eps*delta0:
        #alpha = float(deltanew / float(d.T * (A * d)))
        alpha = float(deltanew / float(np.sum(np.multiply(d, Anfunc(d)))))
        x = x + alpha * d
        r = r - alpha * Anfunc(d)#bn - Anfunc(x)
        deltaold = deltanew
        deltanew = np.linalg.norm(r)
        beta = float(deltanew / float(deltaold))
        d = r + beta * d
        i = i + 1
        print deltanew
    return x
"""
