__author__ = 'mdenil'

import pycuda.autoinit
import pycuda.gpuarray

import generic.optimize.regularizer

class L2Regularizer(generic.optimize.regularizer.L2Regularizer):
    def _sum(self, X):
        return pycuda.gpuarray.sum(X)
