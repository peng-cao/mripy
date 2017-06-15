# this is for fitting of T1 recovery, the three estimates are im(ti) = beta0 - beta1 * exp(-beat2 * ti)
class T1fitting_dataformat:
    def __init__( self, data_shape ):
        self.data_shape = data_shape
        self.beta0      = np.zeros(data_shape,  np.complex128)
        self.beta1      = np.zeros(data_shape,  np.complex128)
        self.beta2      = np.zeros(data_shape,  np.complex128)

    def x2beta( self, x ):
        self.beta0  = x[...,0]
        self.beta1  = x[...,1]
        self.beta2  = x[...,2]
        return self

    def beta2x(self):
        x        = np.zeros(self.data_shape + (3,), np.complex128)
        x[...,0] = self.beta0
        x[...,1] = self.beta1
        x[...,2] = self.beta2
        return x

# this class wrap the Jacobian matrix and transpost of Jacobian matrix into forward and backward operators.
# forward is Jacobian * d_beta, where Jacobian is defined as d_image/d_beta
# backward is Jacobian^T * d_image,
# which in combine with forward function apply to the minimization: min_d_beta ||Jacobian*d_beta-residual||_2^2 + ...
# e.g. for min_d_beta ||J*d_beta-R||_2^2 the d_beta can be acqire as d_beta=(J^H*J)^-1*J^H*R
# e.g. for min_d_beta ||J*d_beta-R||_2^2 + ||beta+d_beta||_1 the d_beta could be solved by CGD, ADMM, IST methods
class T1fitting_opt:
    def __init__( self, TIs ):
        self.TIs        = TIs
        self.x          = None
        self.beta_shape = None

    # shape of each beta map
    def set_beta_shape ( self, shape ):
        self.beta_shape = shape
        return self

    #define x
    def set_x( self, x ):
        self.x = x
        return self

    # Jacobian applies to d_beta,  im = beta0 - beta1 * exp(-beta2 * ti)
    # (ti, beta, d_beta) ---> d_ksp which
    # which is J * d_beta = (d_im/d_beta0)*d_beta0 + (d_im/d_beta1)*d_beta1 + (d_im/d_beta2) * d_beta2
    # d_im/d_beta0 = 1
    # d_im/d_beta1 = exp(-beta2 * ti)
    # d_im/d_beta2 = -beta1 * exp(-beta2 * ti) * (-ti) = beta1 * ti * exp(-beta2 * ti)
    # J*d_beta = d_beta0 + d_beta1 * exp(-bet2 * ti) + d_beta2 * beta1 * ti * exp(-beta2 * ti)
    #          = (d_beta0 + d_beta1 + d_beta2 * beta1 * ti) * exp(-beta2 * ti)
    def forward( self, d_x ):
        if self.beta_shape is None:#if beta_shape is not defined copy the dimenstion from d_x, removing the last dim
            self.beta_shape = d_x.shape[0:len(d_x.shape)-1]
        #beta is estimate
        beta = T1fitting_dataformat(self.beta_shape) #class convert data format
        beta.x2beta( self.x ) #read x
        d_beta = T1fitting_dataformat(self.beta_shape)
        d_beta.x2beta( d_x )   #read dx
        d_im = np.zeros(beta.water.shape + (len(self.TIs),), np.complex128) #image with additional te dim
        for j in range(len(self.TIs)): #loop through ti
            ti          = self.TIs[j]#precompute complex constant
            E1          = np.exp(-ti * beta.beta2) #precompute exponential
            I1          = d_beta.beta0 + d_beta.beta1 + np.multiply(d_beta.beta2, beta.beta1 * ti)#(d_beta0 + d_beta1 + d_beta2 * beta1 * ti)
            d_im[...,j] = np.multiply(I1 , E1) #J*d_beta
        return d_im

    # tanspose of Jacobian applies to d_image
    # d_im = (ti, beta, d_im)--->d_beta
    # im = beta0 - beta1 * exp(-beta2 * ti)
    # => d_im/d_beta0 = 1 => d_im*conj(d_im/d_beta0) = d_im
    # => d_im/d_beta1 = exp(-beta2 * ti) => d_im * conj(d_im/d_beta1) = d_im * conj(exp(-beta2 * ti))
    # => d_im/d_beta2 = beta1 * ti * exp(-beta2 * ti) => d_im*conj(d_im/d_beta2) = d_im * conj(beta1 * ti * exp(-beta2 * ti))
    def backward( self, d_im ):
        if self.beta_shape is None: #beta_shape is not defined, copy d_im dims remove the last dim which is TE dim
            self.beta_shape = d_im.shape[0:len(self.x.shape)-1]
        beta   = T1fitting_dataformat(self.beta_shape) #class defines the beta/data_format
        beta.x2beta(self.x) #convert self.x to beta format
        d_beta = T1fitting_dataformat(self.beta_shape) #claim a zeros data
        for j in range(len(self.TIs)):#loop through TEs
            ti   = self.TIs[j] #precompute a complex constant
            E1   = np.exp(-ti * beta.beta2) #precompute exponential
            E2   = np.multiply(beta.beta1, ti * E1)
            d_beta.beta0  += d_im[...,j]
            d_beta.beta1  += np.multiply(np.conj(E1), d_im[...,j]) #d_im * conj(exp(-beta2 * ti))
            d_beta.beta2  += np.multiply(np.conj(E2), d_im[...,j]) #d_im * conj(beta1 * ti * exp(-beta2 * ti))
        return d_beta.beta2x()#convert to x format

    # apply the model in image space, im = beta0 - beta1 * exp(-beta2 * ti)
    def model( self ):
    	if self.beta_shape is None:
    		self.beta_shape = self.x.shape[0:len(self.x.shape)-1]
        beta   = T1fitting_dataformat(self.beta_shape) #class defines data_format
        beta.x2beta(self.x) #convert self.x to beta format
        im = np.zeros(self.beta_shape + (len(self.TIs),), np.complex128) #image
        for j in range(len(self.TIs)): #loop through TEs
            ti        = self.TIs[j]
            im[...,j] = beta.beta0 - np.multiply(beta.beta1, np.exp( -ti * beta.beta2))
        return im
