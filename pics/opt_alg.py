import numpy as np

"""
back tracking line search from
https://gist.github.com/jiahao/1561144
http://www4.ncsu.edu/~kksivara/ma706/programs/linesearch.m
"""
def BacktrackingLineSearch( f, df, x, p, c = 0.0001, rho = 0.2, ls_Nite = 10 ):
    """
    Backtracking linesearch
    f: function
    x: current point
    p: direction of search
    df: gradient of f at x
    """
    #derphi = np.real(np.sum(np.multiply(p, df(x))))
    #derphi = np.real(np.dot(p.flatten(),df(x)).flatten())
    # for complex vector dot product, conj may be needed
    #
    derphi = np.real(np.dot(p.flatten(),np.conj(df(x)).flatten())) 
    #print("m %g" % derphi)
    f0 = f(x)
    alphak = 1.0
    f_try  = f(x + alphak * p)
    i = 0
    #Loop
    while i < ls_Nite and f_try-f0 >  c * alphak * derphi and f_try > f0 :
        alphak = alphak * rho 
        f_try  = f(x + alphak * p)   
        i += 1
        #print("f(x+ap)-f(x) %g" % (f_try-f0))
    #print(i)
    return alphak, i

"""
back tracking line search from
wiki version
and sparse MRI's implementation
"""
def BacktrackingLineSearch2( f, df, x, p, c = 0.0001, rho = 0.2, ls_Nite = 10 ):
    """
    Backtracking linesearch
    f: function
    x: current point
    p: direction of search
    df: gradient of f at x
    """
    derphi = np.absolute(np.sum(np.multiply(p, df(x))))
    print("m %g" % derphi)
    f0 = f(x)
    alphak = 1.0
    f_try  = f(x + alphak * p)     
    i = 0

    #Loop
    while i < ls_Nite and f_try - f0 >  - c * alphak * derphi and f_try > f0:
        alphak = alphak * rho
        f_try  = f(x + alphak * p) 
        i += 1
        print("f(x+ap)-f(x) %g" % (f(x + alphak * p)-f(x)))        
    #if i > 1:
    #    alphak = alphak * rho
    #if i < 1:
    #    alphak = alphak / rho

    return alphak, i

"""
gradient descent, more general than the ones in proximal_func.py
minimize function f(x) and gradient is df(x)
"""
def gradient_descent( df, x0, Nite, step ):
    x      = x0#np.zeros(x0.shape)    
    eps    = 0.001 #stop criterion
    r      = -df(x)#zero as intial guess #-2*invAfunc(Afunc(x0)-b)#x=x0 as intial guess, i.e. here r=df(x0)
    delta  = np.linalg.norm(r)
    delta0 = delta
    i      = 0 #iteration index
    # iteration
    while i < Nite and delta > eps*delta0:
        r     = -df(x)
        x     = x + step*r
        delta = np.linalg.norm(r)
        #print('gradient %g' % delta) 
        i     = i + 1
    return x
"""
def gradient_descent2( f, df, x0, Nite ):
    x      = x0#np.zeros(x0.shape)    
    eps    = 0.001 #stop criterion
    r      = -df(x)#zero as intial guess #-2*invAfunc(Afunc(x0)-b)#x=x0 as intial guess, i.e. here r=df(x0)
    alpha,nstp = BacktrackingLineSearch(f, df, x, r)
    x = x + alpha * r
    delta  = np.linalg.norm(r)
    delta0 = delta
    i      = 0 #iteration index
    # iteration
    while i < Nite and delta > eps*delta0:
        r     = -df(x)
        alpha,nstp = BacktrackingLineSearch(f, df, x, r)
        x     = x + nstp*r
        delta = np.linalg.norm(r)
        #print('gradient %g' % delta) 
        i     = i + 1
    return x
"""
"""
nonlinear conjugate gradient, more general than the ones in proximal_func.py
minimize function f(x) and gradient is df(x)

"""
def conjugate_gradient( f, df, x0, Nite, ls_Nite = 10 ):
    x = x0#np.zeros(df(x0).shape)
    eps = 0.001
    i = 0
    dx = -df(x) # first step is in the steepest gradient
    #alpha linear search argmin_alpha f(x0 + alpha*dx)
    alpha,nstp = BacktrackingLineSearch(f, df, x, dx, ls_Nite=ls_Nite)
    x = x + alpha * dx
    s = dx
    delta0 = np.linalg.norm(dx)
    deltanew = delta0
    # iteration
    while i < Nite and deltanew > eps*delta0:#and nstp < 20
        dx = -df(x)#-2*invAfunc(Afunc(x)-b)-rho*(x-x0) #this just -df(x)
        #Fletcher-Reeves: beta = np.linalg.norm(dx)/np.linalg.norm(dx_old)
        deltaold = deltanew
        deltanew = np.linalg.norm(dx)#np.sum(dx**2)#
        beta = deltanew / deltaold
        s = dx + beta * s
        #alpha linear search argmin_alpha f(x + alpha*s)
        alpha,nstp = BacktrackingLineSearch(f, df, x, s, ls_Nite=ls_Nite)
        x = x + alpha * s
        i = i + 1
        #print(deltanew)
        #print(nstp)
    return x

"""
based on function gradObj = gOBJ(x,params) in  Miki's sparse MRI
% computes the gradient of the data consistency
    gradObj = params.FT'*(params.FT*x - params.data);
gradObj = 2*gradObj;
"""
def grad_fidelity(FTm_opt, x, b ):
    return 2.0*FTm_opt.backward(FTm_opt.forward(x) - b)

def obj_fidelity(FTm_opt, x, b):
    Tx = FTm_opt.forward(x) - b
    return np.linalg.norm(FTm_opt.forward(x)-b)**2.0

"""
based on function grad = gWT(x,params) in  Miki's sparse MRI
% compute gradient of the L1 transform operator
p = params.pNorm;
WTx = params.WT*x;
G = p*WTx.*(WTx.*conj(WTx)+params.l1Smooth).^(p/2-1);
grad = params.WT'*G; 
"""
def grad_sparsity(Tsparse_opt, x, norm_p=1.0, l1smooth = 5e-1):
    Tx = Tsparse_opt.backward(x)#transfer to sparse space
    G = norm_p*np.divide((Tx), ((np.multiply(Tx, np.conj(Tx)))+l1smooth)**(norm_p/2.0))
    return Tsparse_opt.forward(G)

def obj_sparsity(Tsparse_opt, x, norm_p=1.0, l1smooth = 5e-1):
    Tx =  Tsparse_opt.backward(x)
    return np.sum(((np.multiply(Tx, np.conj(Tx)))+l1smooth)**(norm_p/2.0))
"""
guass newton method
inspired by python code on https://github.com/basil-conto/gauss-newton/blob/master/gaussnewton.py
jacobian is J, pinv(J) = (J^T*J)^-1*J^T 
residual is y - f(t,b), 
(y,t)    is raw data
b beta   is parameter
and aim  is fitting of y=f(t,b)
##############
for minimization problem
min ||f(t,beta) - y||_2^2
solver is below:
within ieration
at beta0, approtimate: f(t,beta) =  f(t,beta0) + J(t,beta0)*(beta-beta0) = f(t,beta0) + J*db,
 and db = beta-beta0 and J = J(t, beta0)
minimization problem become
min ||J*db+f(t,beta0)-y||_2^2, define y-f(t,beta0)=residual
minimization can be rewrite as 
min ||J*db-residual||_2^2
then the db that minimize the last cost function is
db = (J^H*J)^-1*J^H*residual = pinv(J)*residual, and beta = db + beta0
set new beta0 = beta, repeat...
################
min ||f(t,beta) - y||_2^2 + rho*||beta - beta_ref||_2^2
solver is below:
within ieration
at beta0, approtimate: f(t,beta) =  f(t,beta0) + J(t,beta0)*(beta-beta0) = f(t,beta0) + J*db,
 and db = beta-beta0 and J = J(t, beta0)
minimization problem become
min ||J*db+f(t,beta0)-y||_2^2 + rho*||db +beta0 - beta_ref||_2^2, define y-f(t,beta0)=residual
minimization can be rewrite as 
min ||J*db-residual||_2^2 + rho*||db +beta0 - beta_ref||_2^2
then the db that minimize the last cost function is
db = (J^H*J + rho*I)^-1*(J^H*residual-rho*(beta0-beta_ref)), and beta = db + beta0
set new beta0 = beta, repeat...
"""
def guass_newton( jacobian, residual, y, t, beta, Nite, step = 1.0 ):
    eps    = 0.001 #stop criterion
    def dbeta(iy, it, ibeta):# pinv(jacobian) * (residual)
        return np.dot(np.linalg.pinv(jacobian(it,ibeta)),residual(iy, it,ibeta))
    db     = dbeta(y, t, beta)
    nd     = np.linalg.norm(db)
    nd0    = nd
    i      = 0 #iteration indet
    # iteration
    while i < Nite and nd > eps*nd0:
        db    = dbeta(y, t, beta) 
        beta  = beta + step*db
        nd    = np.linalg.norm(db)
        #print('norm of d %g' % nd) 
        i     = i + 1
    return beta

# an etample for defining the f(t,beta) model
class guass_newton_model:
    def __init___( self, y, t, func, Jaco_func ):
        self.y         = y
        self.t         = t
        self.func      = func #f(t,beta)
        self.Jaco_func = Jaco_func #d_f(t,beta)/d_beta, derivitive over beta
    def jacobian( self, ibeta ):
        #compute jacobian matrix, J(ibeta), J = d_f(t,beta)/d_beta, t is constant
        J = self.Jaco_func(ibeta)
        return J
    def residual( self, ibeta ):
        #compute residual, r = y-f(t,beta)
        r = self.y - self.func(self.t, ibeta)
        return r
    #min_beta ||f(t,beta) - y||_2^2
    def dbeta ( self, ibeta ):
        #db=(J^H*J)^-1*J^H*residual=pinv(J)*residual
        return np.dot(np.linalg.pinv(self.jacobian(ibeta)),self.residual(ibeta))
    #min_beta ||f(t,beta) - y||_2^2 + rho*||beta - beta_ref||_2^2
    def prox_dbeta ( self, ibeta, beta_ref, rho ):
        #db=(J^H*J + rho*I)^-1*(J^H*residual-rho*(beta0-beta_ref)),beta0 = ibeta
        J      = self.jacobian(ibeta)
        JHinv  = np.linalg.inv(np.dot(J.H,J) + rho * eye(len(self.x)))
        Hres_b = np.dot(J.H,self.residual)   - rho * (ibeta-beta_ref)
        return np.dot(JHinv,Hres_b)


# use model defined above as input for jacobian and residual functions
# as model_dbeta = guass_newton_model.dbeta
def guass_newton2( model_dbeta, beta, Nite, step = 1.0 ):
    eps    = 0.001 #stop criterion
    db     = model_dbeta(beta)
    nd     = np.linalg.norm(db)
    nd0    = nd
    i      = 0 #iteration index
    # iteration
    while i < Nite and nd > eps*nd0:
        db    = model_dbeta(beta) 
        beta  = beta + step*db
        nd    = np.linalg.norm(db)
        #print('norm of d %g' % nd) 
        i     = i + 1
    return beta