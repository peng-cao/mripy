from multiprocessing import Pool, cpu_count
from functools import partial
#from joblib import Parallel, delayed
def fx(x):
    x = x*2
    return x

def par_wrap(args):
    return fx2(*args)

def fx2(x,y):
    return x+y

#do parallel for function f, f only have one argument, which must be the index of loop
def par_for(f, seqx, use_num_cores = 1):
    if use_num_cores is 1:#use all the cpu cores available
        use_num_cores = cpu_count()
    pool = Pool(use_num_cores)
    y = pool.map(f, seqx)
    pool.close()
    pool.join()
    return y

# a wrap for Pool as an attribute to function, which could have mulitple inputs
# this is from demo code on https://www.binpress.com/tutorial/simple-python-parallelism/121

def parallel(f):
    def parallize(f, seqx):
        return par_for(f, seqx)#f(seqx)#cleaned
    return partial(parallize, f)

def some_function_parallel(x):
    in_function = parallel(fx)
    return in_function(x)

def test():
    #method 1
    #print(par_for(fx, range(10)))
    #method 2
    #par_function = parallel(fx)
    #y = par_function(range(15))
    #print(y)
    #method 3
    #print(some_function_parallel(range(10)))

if __name__ == '__main__':
    test()