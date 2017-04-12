from multiprocessing import Pool
from joblib import Parallel, delayed
def f(x):
    return x

if __name__ == '__main__':
#    p = Pool(5)
#    print(p.map(f, 3))
    inputs = range(20)
    def processFunc(i):
        return i
    test = Parallel(n_jobs=16, verbose=5)(delayed(processFunc)(i) for i in inputs)

