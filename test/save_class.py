"""
save/load class
usage:
import save_class as sc
my = sc.someClass()
my.set()
my.save('myclass')
my.load('myclass')
your = sc.someClass()
your.load('myclass')
your.a


"""



import cPickle
import traceback
import h5py
import numpy as np
class someClass:
    def __init__(self):
        self.a = np.zeros(5)

    def set(self):
        self.a = np.ones(5);
        return self

    def save(self,name):
        """save class as self.name.txt"""
        file = open(name+'.save','w')
        file.write(cPickle.dumps(self.__dict__))
        file.close()
        return self

    def load(self,name):
        """try load self.name.txt"""
        file = open(name+'.save','r')
        dataPickle = file.read()
        file.close()
        self.__dict__ = cPickle.loads(dataPickle)
        return self

