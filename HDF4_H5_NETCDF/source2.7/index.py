import ctypes
import os
path = os.path.dirname(os.path.realpath(__file__))
ctypes.CDLL(os.path.join(path, 'local/lib/libdf.so'),mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL(os.path.join(path, 'local/lib/libmfhdf.so'),mode=ctypes.RTLD_GLOBAL)
from pyhdf.SD import *
import h5py
from netCDF4 import Dataset
def handler(event, context):
    return 0