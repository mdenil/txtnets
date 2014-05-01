__author__ = 'mdenil'

import numpy as np

import pycuda.autoinit
import pycuda.gpuarray

import generic.optimize.regularizer

class L2Regularizer(generic.optimize.regularizer.L2Regularizer):
    def _sum(self, X):
        if isinstance(X, pycuda.gpuarray.GPUArray):
            return pycuda.gpuarray.sum(X)
        else:
            return np.sum(X)
