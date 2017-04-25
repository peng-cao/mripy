#import nufft.nufft_func_cuda as nufft_func_cuda
#nufft_func_cuda.test()

#import test.MRI_recon.test_uterecon as test_uterecon
#test_uterecon.test()

#import test.CS_MRI.cs_IST_1 as cs_IST_1
#cs_IST_1.test()

#import test.CS_MRI.cs_IST_2 as cs_IST_2
#cs_IST_2.test()

#import test.CS_MRI.cs_ADMM as cs_ADMM
#cs_ADMM.test()

#import test.CS_MRI.cs_TV as cs_TV
#cs_TV.test()

#import test.CS_MRI.cs_TV_ADMM_2d as cs_TV_ADMM_2d
#cs_TV_ADMM_2d.test()

#import test.CS_MRI.cs_TV_ADMM_2d_uselib as cs_TV_ADMM_2d_uselib
#cs_TV_ADMM_2d_uselib.test()

#import test.CS_MRI.cs_TV_ADMM_3d as cs_TV_ADMM_3d
#cs_TV_ADMM_3d.test()

#import test.CS_MRI.cs_TV_ADMM_3d_cuda as cs_TV_ADMM_3d_cuda
#cs_TV_ADMM_3d_cuda.test()

#import block_sim.sim_seq_MRF_irssfp_cuda as MRF_irssfp_cuda
#MRF_irssfp_cuda.test()

#import test.class_defines_NNmodel.classmodel_tf_wrap4 as tfmodel
#tfmodel.test1()
#tfmodel.test2()

#import test.numbaCUDA_GPU.test_cufft as test_cufft
#test_cufft.test4()

#import test.espirit.espirit_2d as espirit_2d
#espirit_2d.test()

#import test.espirit.espirit_2d_uselib as espirit_2d_uselib
#espirit_2d_uselib.test()

import test.espirit.espirit_3d_uselib as espirit_3d_uselib
Vim, sim, ft = espirit_3d_uselib.test()