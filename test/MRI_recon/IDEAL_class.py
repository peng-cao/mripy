import numpy as np
import utilities.utilities_func as ut
class IDEAL_dataformat:
    def __init__( self, data_shape ):
    	self.data_shape = data_shape
        self.water      = np.zeros(data_shape,  np.complex128)
        self.fat        = np.zeros(data_shape,  np.complex128)
        self.offres     = np.zeros(data_shape,  np.complex128)

    def x2beta( self, x ):
        self.water  = x[...,0]
        self.fat    = x[...,1]
        self.offres = x[...,2]
        return self

    def beta2x(self):
    	x        = np.zeros(self.data_shape + (3,), np.complex128)
        x[...,0] = self.water
        x[...,1] = self.fat
        x[...,2] = self.offres
        #ut.plotim3(np.absolute(x))
        return x#np.real(x)#.astype(np.float64)
"""
forward based on fwtoolbox_v1_code/doneva/common/@DAMP function dk_data = dforward_op( dx, x, t, f_wf, rel_amp, mask)
%pc guass t is echo time, f_wf is freq seperation between water and fat, 
%dx is d_beta in my defination, x is beta, rel_amp is relative amplitude 
dk_data(:,:,j) = 1i*2*pi*t(j)*(water + fat*np.sum(rel_amp.*exp(1i*2*pi*f_wf*t(j)))).*exp(1i*2*pi*offres*t(j)).*d_offres + ...
        (d_water + ( d_fat*np.sum(rel_amp.*exp(1i*2*pi*f_wf*t(j))))).*exp(1i*2*pi*offres*t(j));
dk_data(:,:,j) = mask(:,:,j).*fft2c(dk_data(:,:,j))

backward based on fwtoolbox_v1_code/doneva/common/@DAMP function dx = dforward_opH(dk_data, x, t, f_wf,rel_amp, mask)
temp     =     ifft2c(mask(:,:,j).*dk_data(:,:,j));                        
d_water  = for all TEs np.sum of  conj(exp(1i*2*pi*offres*t(j))).*temp;         
d_fat    = for all TEs np.sum of (np.sum(rel_amp.*exp(-1i*2*pi*t(j)*f_wf))*conj(exp(1i*2*pi*t(j)*offres))).*temp;             
d_offres = for all TEs np.sum of conj(1i*2*pi*t(j)*exp(1i*2*pi*offres*t(j)).*(water +  np.sum(rel_amp.*exp(1i*2*pi*t(j)*f_wf))*fat)).*temp;  
"""
class IDEAL_opt:
    def __init__( self, TEs, freq_wf, rel_amp ):
        self.TEs        = TEs
        self.freq_wf    = freq_wf #vector freqs for several fat peaks
        self.rel_amp    = rel_amp #relavtive amplitude for several fat peaks
    
    # (t, beta, d_beta) ---> d_ksp which
    # d_ksp=fft(beta--->d_image) 
    # which is J * d_beta = (d_im/d_water)*d_water + (d_im/d_fat)*d_fat + (d_im/d_offres) * d_offres
    # d_im/d_water = exp(Cte * offres)
    # d_im/d_fat = Allfpreak*exp(Cte * offres)
    # d_im/d_offres = water*exp(Cte*offres)*Cte+Allfpeak*fat*Cte*exp(Cte*offres)
    # J*d_beta = (d_water + Allfpeak * d_fat) * exp(Cte*offres)
    #          + (water   + Allfpeak * fat  ) * Cte * exp(Cte*offres) * d_offres
    #          = [(d_water + Allfpeak * d_fat) + (water   + Allfpeak * fat) * Cte * d_offres] * exp(Cte*offres) 
    # define x, and dx in IDEAL_dataformat
    def forward( self, water, fat, offres, d_water, d_fat, d_offres ):
        d_im = np.zeros(water.shape + (len(self.TEs),), np.complex128)
        for j in range(len(self.TEs)):
            Cte         = 1j * 2.0 * np.pi * self.TEs[j]
            Allfpeak    = np.sum(np.multiply(self.rel_amp, np.exp(Cte * self.freq_wf)))
            E1          = np.multiply(water + fat * Allfpeak, d_offres) * Cte
            E2          = d_water + d_fat * Allfpeak
            d_im[...,j] = np.multiply(E1 + E2 , np.exp(Cte * offres))                       
        return d_im

    # tanspose of Jacobian applies to d_image
    # d_im = (t,ifft(d_ksp))--->d_beta
    #offres b0 inhomogenity
    #im = water * exp(Cte * offres)+ ..=> d_im/d_water = exp(Cte * offres) 
    # => d_im*conj(d_im/d_water) = d_im * conjexp(Cte * offres))  
    #im = fat*Allfpeak*exp(Cte*offres) + ..=> d_im/d_fat = Allfpreak*exp(Cte * offres) 
    # => d_im* conj(d_im/d_fat) = d_im * conj(Allfpeak * exp(-Cte*offres))
    #im = water*exp(Cte*offres)+fat*Allfpeak*exp(Cte*offres)
    #     => d_im/d_offres = water*exp(Cte*offres)*Cte+Allfpeak*fat*Cte*exp(Cte*offres)
    #     => d_im*conj(d_im/d_offres) = d_im*conj[exp(Cte*offres)*Cte*(water+Allfpeak*fat)]
    def backward( self, water, fat, offres, d_im ):
        d_water  = np.zeros(water.shape,  np.complex128)
        d_fat    = np.zeros(fat.shape,    np.complex128)
        d_offres = np.zeros(offres.shape, np.complex128)
        for j in range(len(self.TEs)):
            Cte       = 1j * 2.0 * np.pi * self.TEs[j]
            Allfpeak  = np.sum(np.multiply(self.rel_amp, np.exp(Cte * self.freq_wf))) 
            d_water  += np.multiply(np.conj(np.exp(Cte * offres)), d_im[...,j])# d_im * exp(-Cte * offres)   
            d_fat    += np.multiply(np.conj(Allfpeak * np.exp(Cte * offres)), d_im[...,j])#d_im * Allfpeak^-1 * exp(-Cte*offres)
            d_offres += np.multiply(np.conj(np.multiply(Cte * np.exp(Cte * offres),\
                (water + Allfpeak * fat))),d_im[...,j])
        return d_water, d_fat, d_offres

    
    def model( self, water, fat, offres ):
        im = zeros(water.shape + (len(self.TEs),), np.complex)
        for j in range(len(self.TEs)):
            Cte       = 1j * 2.0 * np.pi * self.TEs[j]
            Allfpeak  = np.sum(np.multiply(self.rel_amp, np.exp(Cte * self.freq_wf))) 
            #im = water*exp(Cte*offres)+fat*Allfpeak*exp(Cte*offres)        
            im[...,j] = np.multiply(water, np.exp(Cte * offres)) + Allfpeak * np.multiply(fat, np.exp(Cte * offres))
        return im

# this class wrap the Jacobian matrix and transpost of Jacobian matrix into forward and backward operators.
# forward is Jacobian * d_beta, where Jacobian is defined as d_image/d_beta
# backward is Jacobian^T * d_image, 
# which in combine with forward function apply to the minimization: min_d_beta ||Jacobian*d_beta-residual||_2^2 + ...
# e.g. for min_d_beta ||J*d_beta-R||_2^2 the d_beta can be acqire as d_beta=(J^H*J)^-1*J^H*R
# e.g. for min_d_beta ||J*d_beta-R||_2^2 + ||beta+d_beta||_1 the d_beta could be solved by CGD, ADMM, IST methods
class IDEAL_opt2:
    def __init__( self, TEs, freq_wf, rel_amp ):
        self.TEs        = TEs
        self.freq_wf    = freq_wf #vector freqs for several fat peaks
        self.rel_amp    = rel_amp #relavtive amplitude for several fat peaks
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

    # Jacobian applies to d_beta
    # (t, beta, d_beta) ---> d_ksp which
    # d_ksp=fft(beta--->d_image) 
    # which is J * d_beta = (d_im/d_water)*d_water + (d_im/d_fat)*d_fat + (d_im/d_offres) * d_offres
    # d_im/d_water = exp(Cte * offres)
    # d_im/d_fat = Allfpreak*exp(Cte * offres)
    # d_im/d_offres = water*exp(Cte*offres)*Cte+Allfpeak*fat*Cte*exp(Cte*offres)
    # J*d_beta = (d_water + Allfpeak * d_fat) * exp(Cte*offres)
    #          + (water   + Allfpeak * fat  ) * Cte * exp(Cte*offres) * d_offres
    #          = [(d_water + Allfpeak * d_fat) + (water   + Allfpeak * fat) * Cte * d_offres] * exp(Cte*offres) 
    def forward( self, d_x ):
        if self.beta_shape is None:#if beta_shape is not defined copy the dimenstion from d_x, removing the last dim
            self.beta_shape = d_x.shape[0:len(d_x.shape)-1]
        #beta is estimate: water, fat and offres
        beta = IDEAL_dataformat(self.beta_shape) #class convert data format
        beta.x2beta( self.x ) #read x 
        d_beta = IDEAL_dataformat(self.beta_shape)
        d_beta.x2beta( d_x )   #read dx
        d_im = np.zeros(beta.water.shape + (len(self.TEs),), np.complex128) #image with additional te dim
        for j in range(len(self.TEs)): #loop through te
            Cte         = 1j * 2.0 * np.pi * self.TEs[j]#precompute complex constant
            Allfpeak    = np.sum(np.multiply(self.rel_amp, np.exp(Cte * self.freq_wf)))#np.sum of complex weightings of all peaks
            E1          = np.multiply(beta.water + beta.fat * Allfpeak, d_beta.offres) * Cte#(water + Allfpeak * fat) * Cte * d_offres
            E2          = d_beta.water + d_beta.fat * Allfpeak#(d_water + Allfpeak * d_fat)
            d_im[...,j] = np.multiply(E1 + E2 , np.exp(Cte * beta.offres)) #J*d_beta                  
        return d_im

    # tanspose of Jacobian applies to d_image
    # d_im = (t,ifft(d_ksp))--->d_beta
    #offres b0 inhomogenity
    #im = water * exp(Cte * offres)+ ..=> d_im/d_water = exp(Cte * offres) 
    # => d_im*conj(d_im/d_water) = d_im * conjexp(Cte * offres))  
    #im = fat*Allfpeak*exp(Cte*offres) + ..=> d_im/d_fat = Allfpreak*exp(Cte * offres) 
    # => d_im* conj(d_im/d_fat) = d_im * conj(Allfpeak * exp(-Cte*offres))
    #im = water*exp(Cte*offres)+fat*Allfpeak*exp(Cte*offres)
    #     => d_im/d_offres = water*exp(Cte*offres)*Cte+Allfpeak*fat*Cte*exp(Cte*offres)
    #     => d_im*conj(d_im/d_offres) = d_im*conj[exp(Cte*offres)*Cte*(water+Allfpeak*fat)]
    """
d_water  += conj(exp(1i*2*pi*offres*t(j))).*temp;         
d_fat    += (np.sum(rel_amp.*exp(-1i*2*pi*t(j)*f_wf))*conj(exp(1i*2*pi*t(j)*offres))).*temp;             
d_offres += conj(1i*2*pi*t(j)*exp(1i*2*pi*offres*t(j)).*(water +  np.sum(rel_amp.*exp(1i*2*pi*t(j)*f_wf))*fat)).*temp;      
    """
    def backward( self, d_im ):
        if self.beta_shape is None: #beta_shape is not defined, copy d_im dims
            self.beta_shape = d_im.shape
        beta   = IDEAL_dataformat(self.beta_shape) #class defines the beta/data_format
        beta.x2beta(self.x) #convert self.x to beta format
        d_beta = IDEAL_dataformat(self.beta_shape) #claim a zeros data
        for j in range(len(self.TEs)):#loop through TEs
            Cte       = 1j * 2.0 * np.pi * self.TEs[j] #precompute a complex constant
            Allfpeak  = np.sum(np.multiply(self.rel_amp, np.exp(Cte * self.freq_wf))) #precompute np.sum of complex weights of all fat peaks
            d_beta.water  += np.multiply(np.conj(np.exp(Cte * beta.offres)), d_im[...,j]) #d_im * exp(-Cte * offres)
            d_beta.fat    += np.multiply(np.conj(Allfpeak *np.exp(Cte * beta.offres)), d_im[...,j])
            d_beta.offres += np.multiply(np.conj(np.multiply(Cte * np.exp(Cte * beta.offres),\
            	             (beta.water+Allfpeak*beta.fat))),d_im[...,j])
        return d_beta.beta2x()#convert to x format

    # apply the model in image space, f(x)
    def model( self ):
    	if self.beta_shape is None:
    		self.beta_shape = self.x.shape[0:len(self.x.shape)-1]
        beta   = IDEAL_dataformat(self.beta_shape) #class defines data_format
        beta.x2beta(self.x) #convert self.x to beta format
        im = np.zeros(self.beta_shape + (len(self.TEs),), np.complex128) #image
        for j in range(len(self.TEs)): #loop through TEs
            Cte       = 1j * 2.0 * np.pi * self.TEs[j] #precompute a complex constant
            Allfpeak  = np.sum(np.multiply(self.rel_amp, np.exp(Cte * self.freq_wf))) #np.sum complex weightings of all fat peaks       
            #im = water*exp(Cte*offres)+fat*Allfpeak*exp(Cte*offres)
            im[...,j] = np.multiply(beta.water, np.exp(Cte * beta.offres)) \
                        + Allfpeak * np.multiply(beta.fat, np.exp(Cte * beta.offres))
        return im

    # calculate residual in guass newtown method, this is used in min_d_beta ||J*d_beta-R||_2^2 term
    """
    e.g. for minimization problem
    min ||f(te,beta) - y||_2^2
    solver is below:
    within ieration
    at beta0, approteimate: f(te,beta) =  f(te,beta0) + J(te,beta0)*(beta-beta0) = f(te,beta0) + J*db,
     and db = beta-beta0 and J = J(te, beta0)
    minimization problem become
    min ||J*db+f(te,beta0)-y||_2^2, define y-f(te,beta0)=residual
    minimization can be rewrite as 
    min ||J*db-residual||_2^2
    then the db that minimize the last cost function is
    db = (J^H*J)^-1*J^H*residual = pinv(J)*residual, and beta = db + beta0
    set new beta0 = beta, repeat...  
    for min ||FT*f(te,beta)-y||_2^2  
    FT*f(te,beta) -y = FT*f(te,beta0) + FT*J(te,beta0)*(beta-beta0) -y = Aopt*d_beta-residual
    where Aopt = FT*J, d_beta = beta-beta0, residual = y - FT*f(te,beta0) 
    """
    def residual( self, y_im, FTm = None ):
        #residual = y_im - f(x), both y_im and x should be in image space
        if FTm is not None:
            residual = y_im - FTm.forward(self.model())
        else:
            residual = y_im - self.model()
        return residual

# this opt class can be used in guass-newtown method,
# where minimizing ||x + d_x||_1 over d_x is computed for updating d_x, and x is deta or estimates
# e.g. one could joint wavelet opt with this opt in wavelet L1 minization in CS MRI
class x_add_dx:
    def __init__( self, x = None ):
        self.x = x

    #define x
    def set_x( self, x ):
        self.x = x
        return self
    
    # x_hat-->d_x, d_x = x_hat - x, 
    def forward( self, x_hat ):
        #d_x = x_hat - self.x
        #d_x[:,:,2] = 0.0001*d_x[:,:,2]
        return x_hat - self.x
    
    # d_x-->x_hat, x_hat = x + d_x ,i.e. ||x_hat||_1 = ||x + d_x||_1
    def backward( self, d_x ):
        #x_hat = d_x + self.x
        #x_hat[:,:,2] = 10000*x_hat[:,:,2]
        return d_x + self.x