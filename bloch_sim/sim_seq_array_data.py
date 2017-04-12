from joblib import Parallel, delayed
import multiprocessing
import sim_spin as ss
import numpy as np
import sim_seq as sseq

"""
class simple empty seq simulation and data  output
useage:
import sim_seq_array_data as ssad
a = ssad.seq_arrayin_data(16,100)
a.sim_seq_tc()
a.sim_seq_all()
"""

class seq_arrayin_data:
    'class generate ssfp data from seq simulation'

    def __init__( self, Nexample, Nk ):
        #self.data = 1j*np.zeros((Nexample,Nk))
        self.Nexample = Nexample
        self.Nk = Nk
   
    def sim_seq_all( self ):
        Sall = 1j*np.zeros((self.Nexample,self.Nk))
        for i in range(self.Nexample):
            Sall[i,:] = self.sim_seq_tc(i)  
        return Sall

    def sim_seq_tc( self, i ):
        S = 1j*np.ones(self.Nk)
        return S    

    #simulate MRI signal, with parallel computing
    #def sim_seq_par( self ):
        #inputs = range(self.Nexample)
        #data = Parallel(n_jobs=16, verbose=5)(delayed(self.processInput)(i) for i in inputs)
        #self.data = data
        #return self

    #def results( self ):
        #return self.data
"""
class irssfp seq simulation and data  output
useage:
import sim_seq_array_data as ssad
import numpy as np
M0 = np.matrix([0.0,0.0,1.0]).T 

Nexample = 16
Nk = 960
far = np.pi/10*np.ones(Nk)
far[::2] = -1*far[::2]
trr = 10*np.ones(Nk)

t1t2dfpdr = 0.5*ones((Nk,4))

a = ssad.irssfp_arrayin_data(Nexample,Nk)
a.set(t1t2dfpdr).sim_seq_tc(1,M0, trr, far, ti)#turn one time course, 1 is index, max is Nexample-1
a.set(t1t2dfpdr).sim_seq_all(M0, trr, far ,ti)#turn time courses for all example
"""
class irssfp_arrayin_data( object ):
    'class generate ssfp data from seq simulation'

    def __init__( self, Nexample, Nk ):
        self.T1 = np.zeros(Nexample)
        self.T2 = np.zeros(Nexample)
        self.df = np.zeros(Nexample)
        self.PD = np.zeros(Nexample)
        #self.data = 1j*np.zeros((Nexample,Nk))
        self.Nexample = Nexample
        self.Nk = Nk


    #rescale values
    def set( self, t1t2dfpdr ):
        self.T1 = 5000.0 * t1t2dfpdr[:,0] #ms
        self.T2 = 500.0 * t1t2dfpdr[:,1] #ms
        self.df = 100.0 * (2*t1t2dfpdr[:,2] - 1) #hz
        self.PD = t1t2dfpdr[:,3] #proton density
        return self

    #smulate all time courses
    def sim_seq_all( self, M0, trr, far, ti ):
        Sall = 1j*np.zeros((self.Nexample,self.Nk))
        for i in range(self.Nexample):
            Sall[i,:] = self.sim_seq_tc(i, M0, trr, far, ti )  
        return Sall

    # simulate one time course, i is index < Nexample
    def sim_seq_tc( self, i, M0, trr, far, ti ):
        if self.PD.item(i) > 0.0001 and self.T1.item(i) > 0.0001 and self.T2.item(i) > 0.0001:
            S = sseq.sim_irssfp_arrayin( M0, trr, far, ti, self.T1.item(i), self.T2.item(i), self.df.item(i), self.PD.item(i), self.Nk )
        else:
            S = 1j*np.zeros(self.Nk)
        return S

    #simulate MRI signal, with parallel computing
    #def sim_seq_par( self, Njobs, M0, trr, far, ti ):
        #inputs = range(self.Nexample)
        #num_cores = multiprocessing.cpu_count()
        #data = Parallel(n_jobs=Njobs, verbose=5)(delayed(self.processInput)(i, M0, trr, far, ti ) for i in inputs)
        #self.data = data
        #return self

    #simulate MRI signal, no parallel computing
    #def sim_seq( self, M0, trr, far, ti ):
        #for i in range(self.Nexample):
            #if self.PD.item(i) > 0.0001:
                #self.data[i,:] = sseq.sim_irssfp_arrayin( M0, trr, far, ti, self.T1.item(i), self.T2.item(i), self.df.item(i), self.PD.item(i), self.Nk )
            #else:
                #self.data[i,:] = 1j*np.zeros(self.Nk)
        #return self

    #return the MRI data
    #def results( self ):
        #return self.data
