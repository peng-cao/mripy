#import nufft.nufft_func_cuda as nufft_func_cuda
#nufft_func_cuda.test()

#import test.MRI_recon.test_uterecon as test_uterecon
#test_uterecon.test()

import test.CS_MRI.cs_IST_1 as cs_IST_1
cs_IST_1.test()

import test.CS_MRI.cs_IST_2 as cs_IST_2
cs_IST_2.test()

import test.CS_MRI.cs_ADMM as cs_ADMM
cs_ADMM.test()

import test.CS_MRI.cs_TV as cs_TV
cs_TV.test()

import test.CS_MRI.cs_TV_ADMM as cs_TV_ADMM
cs_TV_ADMM.test()
