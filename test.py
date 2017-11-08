
##########################################################################
# fft function testing
##########################################################################
#import fft.nufft_func_cuda as nufft_func_cuda
#nufft_func_cuda.test()

#import fft.nufft_func as nufft_func
#nufft_func.test()

#import fft.cufft as cufft
#cufft.test5()

#import fft.dft_func as dft_func
#dft_func.test1()
#dft_func.test2()

#import fft.fftw_func as fftw_func
#fftw_func.test1()
#fftw_func.test2()
#fftw_func.test3()

#import test.numbaCUDA_GPU.test_cufft as test_cufft
#test_cufft.test4()

##########################################################################
# MRI regular reconstruction function testing
##########################################################################
#import test.MRI_recon.test_uterecon as test_uterecon
#test_uterecon.test()

#import test.espirit.espirit_2d as espirit_2d
#espirit_2d.test()

#import test.espirit.espirit_2d_uselib as espirit_2d_uselib
#espirit_2d_uselib.test()

#import test.espirit.espirit_3d_uselib as espirit_3d_uselib
#espirit_3d_uselib.test()

#import dwi.dwi_func as dwi_func
#dwi_func.test()

#import low_rank.low_rank_tensor_func as low_rank_tensor_func
#low_rank_tensor_func.test()

##########################################################################
# MRI PICS reconstruction function testing
##########################################################################

#import test.CS_MRI.cs_IST_1 as cs_IST_1
#cs_IST_1.test()

#import test.CS_MRI.cs_IST_2 as cs_IST_2
#cs_IST_2.test()

#import test.CS_MRI.cs_IST_wavelet_L1 as cs_IST_wavelet_L1
#cs_IST_wavelet_L1.test()

#import test.CS_MRI.cs_CGD_wavelet_L1 as cs_CGD_wavelet_L1
#cs_CGD_wavelet_L1.test()


#import test.CS_MRI.pics_IST_3d_wavelet_L1 as pics_IST_3d_wavelet_L1
#pics_IST_3d_wavelet_L1.test()

#import test.CS_MRI.pics_IST_2d_wavelet_L1 as pics_IST_2d_wavelet_L1
#pics_IST_2d_wavelet_L1.test()

#import test.CS_MRI.cs_ADMM as cs_ADMM
#cs_ADMM.test()

#import test.CS_MRI.cs_TV as cs_TV
#cs_TV.test()

#import test.CS_MRI.cs_TV_ADMM_2d as cs_TV_ADMM_2d
#cs_TV_ADMM_2d.test()

#import test.CS_MRI.cs_TV_ADMM_2d_uselib as cs_TV_ADMM_2d_uselib
#cs_TV_ADMM_2d_uselib.test()

#import test.CS_MRI.pics_TV_ADMM_2d_uselib as pics_TV_ADMM_2d_uselib
#pics_TV_ADMM_2d_uselib.test()

#import test.CS_MRI.pics_TV_ADMM_3d_uselib as pics_TV_ADMM_3d_uselib
#pics_TV_ADMM_3d_uselib.test()

#import test.CS_MRI.pics_IST_3d_wavelet_L1_shuffledmprage as pics_IST_3d_wavelet_L1_shuffledmprage
#pics_IST_3d_wavelet_L1_shuffledmprage.test()

#import test.CS_MRI.pics_IST_3dute_wavelet_L1 as pics_IST_3dute_wavelet_L1
#pics_IST_3dute_wavelet_L1.test()

#import test.CS_MRI.pics_IST_3dmultiute_wavelet_L1 as pics_IST_3dmultiute_wavelet_L1
#pics_IST_3dmultiute_wavelet_L1.test()

#import test.CS_MRI.cs_TV_ADMM_3d as cs_TV_ADMM_3d
#cs_TV_ADMM_3d.test()

#import test.CS_MRI.cs_TV_ADMM_3d_cuda as cs_TV_ADMM_3d_cuda
#cs_TV_ADMM_3d_cuda.test()

#import test.CS_MRI.cs_MRF_FC_IST_cuda as cs_MRF_FC_IST_cuda
#cs_MRF_FC_IST_cuda.test1()
#cs_MRF_FC_IST_cuda.test2()

#import test.CS_MRI.cs_MRF_CNN_IST_cuda as cs_MRF_CNN_IST_cuda
#cs_MRF_CNN_IST_cuda.test()

#import test.CS_MRI.cs_MRF_CNN_IST_1p5_cuda as cs_MRF_CNN_IST_1p5_cuda
#cs_MRF_CNN_IST_1p5_cuda.test()

#import test.CS_MRI.cs_MRF_CNN_IST_2_cuda as cs_MRF_CNN_IST_2_cuda
#cs_MRF_CNN_IST_2_cuda.test()

#import test.CS_MRI.CS_IDEAL as CS_IDEAL
#CS_IDEAL.test()

#import test.CS_MRI.CS_IDEAL_ADMMout_GaussNewtonin as CS_IDEAL_ADMMout_GaussNewtonin
#CS_IDEAL_ADMMout_GaussNewtonin.test()

#import test.CS_MRI.CS_IDEAL_CGD as CS_IDEAL_CGD
#CS_IDEAL_CGD.test()

#import test.CS_MRI.CS_IDEAL_myelin_CGD as CS_IDEAL_myelin_CGD
#CS_IDEAL_myelin_CGD.test()

#import test.CS_MRI.CS_IDEAL_myelin_ADMM as CS_IDEAL_myelin_ADMM
#CS_IDEAL_myelin_ADMM.test()

#import test.CS_MRI.CS_IDEAL_waterfat_myelin_CGD as CS_IDEAL_waterfat_myelin_CGD
#CS_IDEAL_waterfat_myelin_CGD.test()

#import test.CS_MRI.L2_IDEAL_FT as L2_IDEAL_FT
#L2_IDEAL_FT.test()

#import test.CS_MRI.L2_IDEAL_orig as L2_IDEAL_orig
#L2_IDEAL_orig.test()

##########################################################################
# MRI sequence simulation testing
##########################################################################
#import test.seq_sim.MRF_test_cuda as MRF_test_cuda
#MRF_test_cuda.test()


#import bloch_sim.sim_seq_MRF_irssfp_cuda as MRF_irssfp_cuda
#MRF_irssfp_cuda.test()

#import test.seq_sim.ssfp_test7_db2 as ssfp_test7_db2
#ssfp_test7_db2.test()


##########################################################################
# neural network testing
##########################################################################

#import test.class_defines_NNmodel.classmodel_tf_wrap4 as tfmodel
#tfmodel.test1()
#tfmodel.test2()


#import neural_network.zoo.tf_wrap_fc as tf_wrap_fc
#tf_wrap_fc.test1()
#tf_wrap_fc.test2()

#import neural_network.zoo.tf_wrap_cnn as tf_wrap_cnn
#tf_wrap_cnn.test1()
#tf_wrap_cnn.test2()

#import neural_network.zoo.tf_wrap_cnn2d as tf_wrap_cnn2d
#tf_wrap_cnn2d.test1()

#import neural_network.zoo.tf_wrap_cnn2d_conv_deconv as tf_wrap_cnn2d_conv_deconv
#tf_wrap_cnn2d_conv_deconv.test1()

#import neural_network.zoo.tf_wrap_cnn2d_Unet as tf_wrap_cnn2d_Unet
#tf_wrap_cnn2d_Unet.test1()

#import neural_network.zoo.tf_wrap_cnn2d_FCN as tf_wrap_cnn2d_FCN
#tf_wrap_cnn2d_FCN.test1()


#import test.neural_network_training.enc_decoder_t1t2b0 as enc_decoder_t1t2b0
#enc_decoder_t1t2b0.test()

#import test.neural_network_training.fc_decoder_t1t2b0 as fc_decoder_t1t2b0python
#fc_decoder_t1t2b0.test()

#import test.neural_network_training.fc_encoder_t1t2b0 as fc_encoder_t1t2b0
#fc_encoder_t1t2b0.test()

#import test.neural_network_training.tfwrap_fc_encoder_t1t2b0 as tfwrap_fc_encoder_t1t2b0
#tfwrap_fc_encoder_t1t2b0.test()

#import test.neural_network_training.tfwrap_fc_encoder_t1t2b0_randomfartrr as tfwrap_fc_encoder_t1t2b0_randomfartrr
#tfwrap_fc_encoder_t1t2b0_randomfartrr.test1()
#tfwrap_fc_encoder_t1t2b0_randomfartrr.test2()
#tfwrap_fc_encoder_t1t2b0_randomfartrr.test3()
#tfwrap_fc_encoder_t1t2b0_randomfartrr.test4()

#import test.neural_network_training.tfwrap_cnn_encoder_t1t2b0_randomfartrr as tfwrap_cnn_encoder_t1t2b0_randomfartrr
#tfwrap_cnn_encoder_t1t2b0_randomfartrr.test1()
#tfwrap_cnn_encoder_t1t2b0_randomfartrr.test2()

#import test.neural_network_training.tf_wrap_cnn2d_Unet_lung_kaggle as tf_wrap_cnn2d_Unet_lung_kaggle
#tf_wrap_cnn2d_Unet_lung_kaggle.test1()

#import test.neural_network_training.tf_wrap_cnn2d_Unet_heart as tf_wrap_cnn2d_Unet_heart
#tf_wrap_cnn2d_Unet_heart.test1()


#import test.neural_network_training.gan_example as gan_example
#gan_example.test1()

#import neural_network.zoo.tf_wrap_fc_GAN as tf_wrap_fc_GAN
#tf_wrap_fc_GAN.test1()

#import test.neural_network_training.tf_wrap_fc_GAN_MNIST as tf_wrap_fc_GAN_MNIST
#tf_wrap_fc_GAN_MNIST.test1()

#import test.neural_network_training.tfwrap_fc_jing_dict as tfwrap_fc_jing_dict
#tfwrap_fc_jing_dict.test2()

#import test.neural_network_training.tfwrap_cnn_jing_dict as tfwrap_cnn_jing_dict
#tfwrap_cnn_jing_dict.test1()
#tfwrap_cnn_jing_dict.test2()

#import test.neural_network_training.tfwrap_fc_jing_randt1t2 as tfwrap_fc_jing_randt1t2
#tfwrap_fc_jing_randt1t2.test1()
#tfwrap_fc_jing_randt1t2.test2()

#import test.neural_network_training.tfwrap_fc_jing_randt1t2_segment as tfwrap_fc_jing_randt1t2_segment
#tfwrap_fc_jing_randt1t2_segment.test1()
#tfwrap_fc_jing_randt1t2_segment.test2()
##########################################################################
# parallel testing
##########################################################################
#import test.parallel_compute_multiCPU.blas_test as blas_test
#blas_test.test()
