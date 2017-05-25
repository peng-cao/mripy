import numpy as np
from time import time

#do timing for the code
class timing():
    def __intial__( self ):
        self.time = 0.0
        self.t0   = 0.0
        self.t1   = 0.0
        #self.listtime = []

    def start( self ):
        self.t0   = time()
        self.t1   = self.t0
        return self

    def stop( self ):
        self.t1   = time()
        self.time = self.t1 - self.t0
        return self

    def gettime( self ):
        return self.time

    def display( self, str='' ):
        print( str + 'Executime time: %g sec' % self.time) 
        return self