__author__ = 'mdenil'

import pycuda.gpuarray

import generic.model.model

import gpu.space
import gpu.model.layer


class CSM(generic.model.model.CSM, gpu.model.layer.Layer):
    Space = gpu.space.GPUSpace
    NDArray = pycuda.gpuarray.GPUArray

    def move_to_cpu(self):
        from gpu.model.host_device_component_mapping import get_cpu_analog
        cpu_class = get_cpu_analog(self.__class__)

        return cpu_class(layers=[l.move_to_cpu() for l in self.layers])