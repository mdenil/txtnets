__author__ = 'mdenil'

import pycuda.autoinit
import pycuda.gpuarray
import pycuda.cumath

import generic.optimize.sgd

class SGD(generic.optimize.sgd.SGD):
    def _mean_abs(self, x):
        return pycuda.gpuarray.sum(pycuda.cumath.fabs(x)) / float(x.size)
