__author__ = 'mdenil'

import numpy as np
import pycuda.autoinit
import pycuda

import gpu.model.layer
import generic.model.repeat_layer

class RepeatLayer(generic.model.repeat_layer.RepeatLayer, gpu.model.layer.Layer):
    def _zeros_like(self, x):
        if isinstance(x, pycuda.gpuarray.GPUArray):
            return pycuda.gpuarray.zeros_like(x)
        else:
            return np.zeros_like(x)