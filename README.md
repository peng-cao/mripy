# mripy
A python based MRI reconstruction toolbox with compressed sensing, parallel imaging and machine-learning functions
-----"bloch_sim/"      contains functions for MRI sequence simuation, this is designed for MR fingerprinting experiment
-----"nufft/"          contains functions for non uniform FFT, implemented for both CPU and GPU
-----"fft/"            this is a wrap of cuda fft, i.e. cufft
-----"pics/"           contains optimization algorithms, such as ADMM, conjugate gradient, gradient descent, for MRI compressed sensing and parallel imaging reconstructions, as well as operators such as total variation, Hankel matrix
-----"neural_network/" contains a wrap of tensorflow for creating and testing neural_network
-----"test/"           contains testing code for above functions and something underdevelop right now

