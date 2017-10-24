import utilities.utilities_class as utc
import numba.cuda as cuda
from multiprocessing import cpu_count
#e.g. debug_print.atlevel(1,'hello debug world! ', 21).global_level()
debug_print = utc.debug_level(5)# global_set_debug_level = 0 
#gpu_list    = cuda.cudadrv.devices._DeviceList()
gpu_available = cuda.is_available()
cpu_count     = cpu_count()

