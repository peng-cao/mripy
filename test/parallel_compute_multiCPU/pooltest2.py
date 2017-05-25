def multi_run_wrapper(args):
   return add(*args)
def add(x,y):
    return x+y
if __name__ == "__main__":
    from multiprocessing import Pool
    pool = Pool(4)
    results = pool.map(multi_run_wrapper,[(1,2),(2,3),(3,4)])
    print results
