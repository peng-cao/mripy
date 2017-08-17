## mripy
A python based MRI reconstruction toolbox with compressed sensing, parallel imaging and machine-learning functions
## Key functions
* "bloch_sim/"      contains functions for MRI sequence simulation, these functions are designed for MR fingerprinting experiment
* "fft/"            this is a wrap of FFT functions, i.e. cuFFT, FFTW, and NUFFT, implemented for both CPU and GPU
* "pics/"           contains optimization algorithms, such as ADMM, conjugate gradient, gradient descent, for MRI compressed sensing and parallel imaging reconstructions, as well as operators such as total variation, Hankel matrix, coil sensitivity
* "neural_network/" contains a wrap of tensorflow functions for creating and testing neural_network, and zoo/ contains examples for full connection net, CNN, Unet, and FCN.  
* "test/"           contains testing code for above functions and something I am working on right now, e.g. MRI PICS reconstruction, IDEAL + CS reconstruction, FC or CNN for MRF quantification, Unet for creating mask on medical images
