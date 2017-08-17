from joblib import Parallel, delayed
import multiprocessing
import sim_spin as ss
import numpy as np

"""
function simulate ssfp sequence with fixed tr, fa, and phase cycling

usage: 
import sim_seq as sseq
import numpy as np
M0 = np.matrix([0.0,0.0,1.0]).T
tr = 10.0 #ms
fa = np.pi/4
T1 = 1000.0 #ms
T2 = 200.0 #ms
df = -40.0
PD = 1.0 #proton density
Nk = 960
S = sseq.sim_ssfp( M0, tr, fa, T1, T2, df, PD, Nk )
print abs(S)
"""
def sim_ssfp( M0, tr, fa, T1, T2, df, PD, Nk ):
    "simulate ssfp sequence"
    S = 1j*np.zeros(Nk) #initial an array
    M = M0*PD
    for k in range(Nk):
        #excitation
        M = ss.throt(np.absolute(fa), np.angle(fa))*M
        #free precession half TR
        A,B = ss.freeprecess(tr/2.0,T1,T2,df)
        M = A*M+PD*B.T
        #save the MR signal
        S.itemset(k, M.item(0)+1j*M.item(1))
        #second half of TR
        M = A*M+PD*B.T
        fa = -fa;
    return S

"""
function simulate ssfp sequence with array inputs of tr and fa

usage:
import sim_seq as sseq
import numpy as np
far = np.pi/10*np.ones(960)
far[::2] = -1*far[::2]
trr = 10*np.ones(960)
M0 = np.matrix([0.0,0.0,-1.0]).T 
T1 = 1000.0 #ms
T2 = 200.0 #ms
df = 0.0
PD = 1.0 #proton density
Nk = 960
S = sseq.sim_ssfp_arrayin( M0, tr, fa, T1, T2, df, PD, Nk )
print abs(S)
"""
def sim_ssfp_arrayin( M0, trr, far, T1, T2, df, PD, Nk ):
    "simulate ssfp sequence"
    S = 1j*np.zeros(Nk) #initial an array
    M = M0*PD
    for k in range(Nk):
        #excitation
        M = ss.throt(np.absolute(far.item(k)), np.angle(far.item(k)))*M
        #free precession half TR
        A,B = ss.freeprecess(trr.item(k)/2.0,T1,T2,df)
        M = A*M+PD*B.T
        #save the MR signal
        S.itemset(k, M.item(0)+1j*M.item(1))
        #second half of TR
        M = A*M+PD*B.T
    return S

"""
function simulate IR-SSFP sequence with array inputs of tr and fa

usage:
import sim_seq as sseq
import numpy as np
far = np.pi/10*np.ones(960)
far[::2] = -1*far[::2]
trr = 10*np.ones(960)
M0 = np.matrix([0.0,0.0,-1.0]).T 
ti = 5 #ms
T1 = 1000.0 #ms
T2 = 200.0 #ms
df = 0.0
PD = 1.0 #proton density
Nk = 960
S = sseq.sim_irssfp_arrayin( M0, trr, far, ti, T1, T2, df, PD, Nk )
print abs(S)
"""
def sim_irssfp_arrayin( M0, trr, far, ti, T1, T2, df, PD, Nk ):
    "simulate ssfp sequence"
    S = 1j*np.zeros(Nk) #initial an array
    M = M0*PD
    #inversion pulse
    M = ss.throt(np.pi, 0)*M
    #inversion time
    Air,Bir = ss.freeprecess(ti,T1,T2,df)
    M = Air*M+PD*Bir.T
    for k in range(Nk):
        #excitation
        M = ss.throt(np.absolute(far.item(k)), np.angle(far.item(k)))*M
        #free precession half TR
        A,B = ss.freeprecess(trr.item(k)/2.0,T1,T2,df)
        M = A*M+PD*B.T
        #save the MR signal
        S.itemset(k, M.item(0)+1j*M.item(1))
        #second half of TR
        M = A*M+PD*B.T
    return S

"""
function simulate gre sequence with fixed te, tr and fa

usage: 
import sim_seq as sseq
import numpy as np
M0 = np.matrix([0.0,0.0,1.0]).T
tr = 10.0 #ms
te = 2 #ms
fa = np.pi/4
T1 = 1000.0 #ms
T2 = 200.0 #ms
df = -40.0
PD = 1.0 #proton density
Nk = 960
S = sseq.sim_gre( M0, tr, te, fa, T1, T2, df, PD, Nk )
print abs(S)
"""
def sim_gre( M0, tr, te, fa, T1, T2, df, PD, Nk, Ndum = 5 ):
    "simulate gre sequence"
    S = 1j*np.zeros(Nk) #initial an array

    for _ in dummy(Ndum):  #dummy loop
        M = M0*PD        
        for k in range(Nk):
            #excitation
            M = ss.throt(np.absolute(fa), np.angle(fa))*M
            #free precession 0 to TE
            A,B = ss.freeprecess(te,T1,T2,df)
            M = A*M+PD*B.T
            #save the MR signal
            S.itemset(k, (M.item(0)+1j*M.item(1))*exp(-1j*np.angle(fa)))
            #free precession TE to TR
            A,B = ss.freeprecess(tr-te,T1,T2,df)
            M = A*M+PD*B.T
            #gradient spoiler
            M.itemset(0,0);
            M.itemset(1,0);
            #fa = -fa;
    return S
