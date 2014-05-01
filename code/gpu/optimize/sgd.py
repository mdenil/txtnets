__author__ = 'mdenil'

import numpy as np
import pycuda.autoinit
import pycuda.gpuarray
import pycuda.cumath

import generic.optimize.sgd

class SGD(generic.optimize.sgd.SGD):
    def _mean_abs(self, x):
        if isinstance(x, pycuda.gpuarray.GPUArray):
            return pycuda.gpuarray.sum(pycuda.cumath.fabs(x)) / float(x.size)
        else:
            return np.mean(np.abs(x))
