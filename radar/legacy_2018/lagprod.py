import ctypes as C
import numpy as n
import time

# to compile the library: 
#
# gcc -L/usr/lib64  -lfftw3f  -shared -fPIC -O3 -o liblagprod.so lagprod.c
#
# install openblas and pyfftw
#@numba.vectorize([numba.float64(numba.complex128),numba.float32(numba.complex64)])
#def abs2(x):
#    return x.real**2 + x.imag**2

# more efficient c operations

# Stuff to set up access to C routine for median filter (fmedfil)
_fmed = n.ctypeslib.load_library('liblagprod', './Ccode')
_fmed.lagprod.restype = C.c_int
_fmed.lagprod.argtypes = [C.POINTER(C.c_float), C.POINTER(C.c_float),C.POINTER(C.c_float),C.POINTER(C.c_float),C.POINTER(C.c_float),C.POINTER(C.c_float)]
def lagprod(tx0,tx1,rx0,rx1,tr,tt):
    # these are hardocded in the C program to make it compile to faster code (vectorization)
    nrange=5120
    _fmed.lagprod(tx0.ctypes.data_as(C.POINTER(C.c_float)), tx1.ctypes.data_as(C.POINTER(C.c_float)), rx0.ctypes.data_as(C.POINTER(C.c_float)), rx1.ctypes.data_as(C.POINTER(C.c_float)), rt.ctypes.data_as(C.POINTER(C.c_float)), tt.ctypes.data_as(C.POINTER(C.c_float)))
    return(1)
