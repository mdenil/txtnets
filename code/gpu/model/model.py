__author__ = 'mdenil'

import pycuda.gpuarray

import generic.model.model

import gpu.space
import gpu.model.layer

class CSM(generic.model.model.CSM, gpu.model.layer.Layer):
    Space = gpu.space.CPUSpace
    NDArray = pycuda.gpuarray.GPUArray