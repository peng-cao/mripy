import numpy as np
from time import time

# should have this soon
#global_set_debug_level = 0 
#levle 0 print inside/outside function infromation
#level 1 print run timing information
#level 2 print basic parameters information from algrithm
#level 3 print detailed paprameter information
#level 4 print warning, e.g. dimension mismatch

class debug_level:
    def __init__( self, global_level = 0):
        self.global_set_debug_level = global_level

    def atlevel( self, debug_print_level, message, arg = None):
        if debug_print_level <= self.global_set_debug_level:
            if arg is not None:
                print(message + repr(arg))
            else:
                print(message)
        return self

    def global_level( self ):
        print('global debug level: %g' % self.global_set_debug_level)
        return self.global_set_debug_level


#do timing for the code
class timing():
    def __init__( self ):
        self.time = 0.0
        self.t0   = 0.0
        self.t1   = 0.0
        self.str  = ''
        #self.listtime = []

    def start( self, str = '' ):
        if str is not '':
            self.str = str
        else:
            self.str  = ''
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
        if str is not '':
            self.str = str
        print( self.str + 'execute time: %g sec' % self.time) 
        self.str  = ''
        return self