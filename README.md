## mripy
A python based MRI reconstruction toolbox with compressed sensing, parallel imaging and machine-learning functions
## Key functions
* "bloch_sim/"      contains functions for MRI sequence simulation, this is designed for MR fingerprinting experiment
* "fft/"            this is a wrap of cuda fft, i.e. cufft, fftw, and NUFFT, implemented for both CPU and GPU
* "pics/"           contains optimization algorithms, such as ADMM, conjugate gradient, gradient descent, for MRI compressed sensing and parallel imaging reconstructions, as well as operators such as total variation, Hankel matrix, coil sensitivity
* "neural_network/" contains a wrap of tensorflow for creating and testing neural_network, and zoo/ contains examples for full connection, CNN, Unet, and FCN.  
* "test/"           contains testing code for above functions and something I am working on right now
