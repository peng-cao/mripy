import h5py
f = h5py.File('/data/larson/brain_uT2/2016-09-13_3T-volunteer/ute_32echo_random-csreconallec_l2_r0p01.mat')
im = f['imallplus'][0]