import numpy as np
import proximal_func as pf

"""
iterative soft-thresholding
argmin ||Ax-b|||_2^2+(th/2)*||x||_1
matrix A input 
"""
def IST_1( A, b, Nite, step, th ):
    #inverse operator
    invA = np.linalg.pinv(A)
    x_acc = step*(invA.dot(b))#np.zeros(x.shape)
    # iteration
    for _ in range(Nite):
        # soft threshold
        x = pf.prox_l1_soft_thresh(x_acc,th)
        #residual
        r = A.dot(x) - b
        x_acc = x_acc - step*(invA.dot(r))
        print np.linalg.norm(r)
    return x

"""
iterative soft-thresholding
argmin ||A(x)-b|||_2^2+(th/2)*||x||_1
function A() input 
"""
def IST_2( Afunc, invAfunc, b, Nite, step, th ):
    x_acc = step*invAfunc(b)#np.zeros(x.shape)
    # iteration
    for _ in range(Nite):
        # soft threshold
        x = pf.prox_l1_soft_thresh(x_acc, th)
        #residual
        r = Afunc(x) - b
        x_acc = x_acc - step*invAfunc(r)
        print np.linalg.norm(r)
    return x

"""
iterative soft-thresholding
argmin ||A(x)-b|||_2^2+(th/2)*||Tfunc(x)||_1
function A() input 
Tfunc can be total variation, wavelet, singular values of Hankel etc.
"""
def IST_3( Afunc, invAfunc, Tfunc, invTfunc, mask, b, Nite, step, th ):
    x_acc = step*invAfunc(b) #np.zeros(x.shape)
    # iteration
    for _ in range(Nite):
        # soft threshold
        x = pf.prox_l1_Tf_soft_thresh(Tfunc,invTfunc,x_acc, th)
        #residual
        r = Afunc(x,mask) - b
        x_acc = x_acc - step*invAfunc(r)
        print np.linalg.norm(r)
    return x


"""
ADMM for argmin ||Ax-b|||_2^2+lambda*||x||_1
Lagrangian is L(x,z,y) = f(x) + g(z) + y^H(x-z) + (rho/2)*||x-z||_2^2 
for f(x) = ||Ax-b|||_2^2
and g(z) = lambda*||z||_1, 
so that L(x,z,u) = f(x) + g(z) + (rho/2)*||x-z+y/rho||_2^2 + const, 
and u = y/rho 

duality is duality(y) = inf_x,z L(x,z,y)
dual problem is  max duality(y) <= f0+g0 (which is the minimal value of orginal cost function f(x) + g(z))
gradient of duality is grad_dual(y) = x-z
gradient mehtod for dual problem (maximizing dual) is u^k+1 = u^k + alphi*(x^k-z^k), alphi is step size

dual ascent method is
x^k+1 = argmin_x L(x,z^k,u^k) = argmin_x [ f(x) + (rho/2)*||x-z^k+u^k||_2^2 ] = argmin_x [ ||Ax-b|||_2^2 + (rho/2)*||x-z^k+u^k||_2^2 ]
z^k+1 = argmin_z L(x^k,z,u^k) = argmin_z [ g(z) + (rho/2)*||x^k-z+u^k||_2^2 ] = argmin_z [ lambda*||z||_1 + (rho/2)*||x^k-z+u^k||_2^2 ]
u^k+1 = u^k + alphi*(x^k-z^k)

using proximal functions
x^k+1 = prox_l2_Axnb(A,b,x0=z^k-u^k,rho)
z^k+1 = prox_l1_soft_thresh(x0=x^k+u^k,th = lambda/rho)
u^k+1 = u^k + alphi*(x^k+1-z^k+1)
"""
def ADMM_l2Axnb_l1x_1( A, b, Nite, step, l1_r, rho ):
    z = np.pinv(A).dot(b)
    u = np.zeros(z.shape)
    # iteration
    for _ in range(Nite):
        # soft threshol
        x = pf.prox_l2_Axnb(A,b,z-u,rho)
        z = pf.prox_l1_soft_thresh(x+u,l1_r/rho)
        u = u + step*(x-z)
        print np.linalg.norm(x-z)
    return x
"""
faster version with percalculation
"""
def ADMM_l2Axnb_l1x_2( A, b, Nite, step, l1_r, rho ):
    z = np.pinv(A).dot(b)
    u = np.zeros(z.shape)
    Q_dot, A_T_b = prox_l2_Axnb_precomputpart( A, b, rho )
    # iteration
    for _ in range(Nite):
        # soft threshold
        x = Q_dot(A_T_b + rho*(z-u)) #prox_l2_Axnb_iterpart( Q_dot, A_T_b, z-u, rho )
        z = pf.prox_l1_soft_thresh(x+u,l1_r/rho)
        u = u + step*(x-z)
        print np.linalg.norm(x-z)
    return x

"""
ADMM for argmin ||Afunc(x)-b|||_2^2+lambda*||x||_1
Lagrangian is L(x,z,y) = f(x) + g(z) + y^H(x-z) + (rho/2)*||x-z||_2^2 
for f(x) = ||Afunc(x)-b|||_2^2
and g(z) = lambda*||z||_1, 
so that L(x,z,u) = f(x) + g(z) + (rho/2)*||x-z+y/rho||_2^2 + const, 
and u = y/rho 

duality is duality(y) = inf_x,z L(x,z,y)
dual problem is  max duality(y) <= f0+g0 (which is the minimal value of orginal cost function f(x) + g(z))
gradient of duality is grad_dual(y) = x-z
gradient mehtod for dual problem (maximizing dual) is u^k+1 = u^k + alphi*(x^k-z^k), alphi is step size

dual ascent method is
x^k+1 = argmin_x L(x,z^k,u^k) = argmin_x [ f(x) + (rho/2)*||x-z^k+u^k||_2^2 ] = argmin_x [ ||Afunc(x)-b|||_2^2 + (rho/2)*||x-z^k+u^k||_2^2 ]
z^k+1 = argmin_z L(x^k,z,u^k) = argmin_z [ g(z) + (rho/2)*||x^k-z+u^k||_2^2 ] = argmin_z [ lambda*||z||_1 + (rho/2)*||x^k-z+u^k||_2^2 ]
u^k+1 = u^k + alphi*(x^k-z^k)

using proximal functions
x^k+1 = prox_l2_Afxnb_GD(Afunc, invAfunc, b, x0=z^k-u^k, rho, Nite=100, step=0.01 )
z^k+1 = prox_l1_soft_thresh(x0=x^k+u^k,th = lambda/rho)
u^k+1 = u^k + alphi*(x^k+1-z^k+1)
"""
def ADMM_l2Afxnb_l1x( Afunc, invAfunc, b, Nite, step, l1_r, rho ):
    z = invAfunc(b) #np.zeros(x.shape)
    u = np.zeros(z.shape)
    # iteration
    for _ in range(Nite):
        # soft threshold
        #x = pf.prox_l2_Afxnb_GD(Afunc,invAfunc,b,z-u,rho,10,0.1)
        x = pf.prox_l2_Afxnb_CGD( Afunc, invAfunc, b, z-u, rho, 3 )
        z = pf.prox_l1_soft_thresh(x+u,l1_r/rho)
        u = u + step*(x-z)
        print np.linalg.norm(x-z)
    return x

def ADMM_l2Afxnb_l1Tfx( Afunc, invAfunc, Tfunc, invTfunc, b, Nite, step, l1_r, rho ):
    z = invAfunc(b)#np.zeros(x.shape)
    u = np.zeros(z.shape)
    # iteration
    for _ in range(Nite):
        # soft threshold
        x = pf.prox_l2_Afxnb_GD(Afunc,invAfunc,b,z-u,rho,10,0.1)
        z = pf.prox_l1_Tf_soft_thresh(Tfunc,invTfunc,x+u,l1_r/rho)
        u = u + step*(x-z)
        print np.linalg.norm(x-z)
    return x
"""
ADMM for argmin ||Afunc(x_1)-b|||_2^2+lambda1*||x_2||_1 + lambda2*||Tfunc(x_3)||_1
Lagrangian is L(x,z,y) = f_1(x_1) + f_2(x_2) + f_3(x_3) + g(z) + sum_i=1,2,3_[y^H(x_i-z) + (rho/2)*||x_i-z||_2^2 ]
for f_1(x_1) = ||Afunc(x_1)-b||_2^2, f_2(x_2) = lambda1*||x_2||_1, f_3(x_3) = lambda2*||Tfunc(x_3)||_1 and g(z) = 0

so that L(x_i,z,u) = sum_i=1,2,3_( f_i(x_i) + (rho/2)*||x_i-z+u_i||_2^2 ) + const, and u_i = y_i/rho 

gradient mehtod for dual problem (maximizing dual) is u_i^k+1 = u_i^k + alphi*(x_i^k-z^k), alphi is step size

dual ascent method is
x_i^k+1 = argmin_x_i L(x_i,z^k,u^k) = argmin_x_i [ f_i(x_i) + (rho/2)*||x_i-z^k+u_i^k||_2^2 ]
=>x_1^k+1 = argmin_x_1 [ ||Afunc(x_1)-b||_2^2     + (rho/2)*||x_1-z^k+u_1^k||_2^2 ]
=>x_2^k+1 = argmin_x_2 [ lambda1*||x_2||_1        + (rho/2)*||x_2-z^k+u_2^k||_2^2 ]
=>x_3^k+1 = argmin_x_3 [ lambda2*||Tfunc(x_3)||_1 + (rho/2)*||x_3-z^k+u_2^k||_2^2 ]
z^k+1 = argmin_z L(x^k,z,u^k) = argmin_z [(rho/2)*||x^k-z+u^k||_2^2 ] 
=> z^k+1 = average(x_i^k + u_i^k) 
u_i^k+1 = u_i^k + alphi*(x_i^k-z^k)

using proximal functions
x_1^k+1 = prox_l2_Afxnb_GD(Afunc, invAfunc, b,  x0=z^k-u_1^k, rho, Nite=100, step=0.01 )
x_2^k+1 = prox_l1_soft_thresh   (               x0=z^k-u_2^k,th = lambda1/rho)
x_3^k+1 = prox_l1_Tf_soft_thresh(Tfunc,invTfunc,x0=z^k-u_3^k,th = lambda2/rho)
z^k+1 = average(x_i^k)
u_i^k+1 = u_i^k + alphi*(x_i^k+1-z^k+1)
"""
def ADMM_l2Afxnb_l1x_l1Tfx( Afunc, invAfunc, Tfunc, invTfunc, b, Nite, step, l1_r1, L1_r2, rho ):
    z = invAfunc(b)
    u1 = np.zeros(z.shape)
    u2 = np.zeros(z.shape)
    u3 = np.zeros(z.shape)
    # iteration
    for _ in range(Nite):
        # soft threshold
        x1 = pf.prox_l2_Afxnb_GD(Afunc,invAfunc,b,z-u1,rho,10,0.1)
        x2 = pf.prox_l1_soft_thresh(z-u2,l1_r1/rho)
        x3 = pf.prox_l1_Tf_soft_thresh(Tfunc,invTfunc,z-u3,l1_r2/rho)
        z = (x1 + x2 + x3)/3 + (u1 + u2 + u3)/3
        u1 = u1 + step*(x1-z)
        u2 = u2 + step*(x2-z)
        u3 = u3 + step*(x3-z)
        print np.linalg.norm(x1-z)
    return x1

def ADMM_l2Afxnb_l1x_2( Afunc, invAfunc, b, Nite, step, l1_r1, rho ):
    z = invAfunc(b)
    u1 = np.zeros(z.shape)
    u2 = np.zeros(z.shape)
    # iteration
    for _ in range(Nite):
        # soft threshold
        #x1 = pf.prox_l2_Afxnb_GD(Afunc,invAfunc,b,z-u1,rho,10,0.1)
        x1 = pf.prox_l2_Afxnb_CGD( Afunc, invAfunc, b, z-u1, rho, 3 )
        x2 = pf.prox_l1_soft_thresh(z-u2,l1_r1/rho)
        z = (x1 + x2)/2.0 + (u1 + u2)/2.0
        u1 = u1 + step*(x1-z)
        u2 = u2 + step*(x2-z)
        print np.linalg.norm(x2-x1)
    return x1
