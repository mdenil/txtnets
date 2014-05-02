__author__ = 'mdenil'

import numpy as np

import gpu.space
from gpu.model import layer

import gpu.utils

class HostToDevice(layer.Layer):
    def fprop(self, X, meta):
        fprop_state = {}
        Y, meta['space_above'] = gpu.space.GPUSpace.from_cpu(X.astype(np.float32), meta['space_below'])
        return Y, meta, fprop_state

    def bprop(self, delta, meta, fprop_state):
        out, meta['space_below'] = meta['space_above'].to_cpu(delta)
        return out, meta

    def move_to_cpu(self):
        from gpu.model.host_device_component_mapping import get_cpu_analog
        cpu_class = get_cpu_analog(self.__class__)

        return cpu_class()


class DeviceToHost(layer.Layer):
    def fprop(self, X, meta):
        fprop_state = {}
        Y, meta['space_above'] = meta['space_below'].to_cpu(X)
        return Y, meta, fprop_state

    def bprop(self, delta, meta, fprop_state):
        out, meta['space_below'] = gpu.space.GPUSpace.from_cpu(delta.astype(np.float32), meta['space_above'])
        return out, meta

    def move_to_cpu(self):
        from gpu.model.host_device_component_mapping import get_cpu_analog
        cpu_class = get_cpu_analog(self.__class__)

        return cpu_class()
