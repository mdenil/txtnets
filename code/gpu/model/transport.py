__author__ = 'mdenil'

import gpu.space
from gpu.model import layer

import gpu.utils

class CPUToGPU(layer.Layer):
    def fprop(self, X, meta):
        fprop_state = {}
        Y, meta['space_above'] = gpu.space.GPUSpace.from_cpu(X, meta['space_below'])
        return Y, meta, fprop_state

    def bprop(self, delta, meta, fprop_state):
        out, meta['space_below'] = meta['space_above'].to_cpu(delta)
        return out, meta


class GPUToCPU(layer.Layer):
    def fprop(self, X, meta):
        fprop_state = {}
        Y, meta['space_above'] = meta['space_below'].to_cpu(X)
        return Y, meta, fprop_state

    def bprop(self, delta, meta, fprop_state):
        out, meta['space_below'] = gpu.space.GPUSpace.from_cpu(delta, meta['space_above'])
        return out, meta