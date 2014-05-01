__author__ = 'mdenil'

import numpy as np

class Layer(object):
    def __init__(self, *args, **kwargs):
        pass

    def params(self, *args, **kwargs):
        return []

    def grads(self, *args, **kwargs):
        return []

    def pack(self):
        # TODO: everything on the device
        return np.concatenate([p.get().ravel() for p in self.params()])

    def unpack(self, values):
        # TODO: everything on the device
        start = 0
        for param in self.params():
            end = start + param.size
            param.set(values[start:end].reshape(param.shape).astype(param.dtype))
            start = end

    def __repr__(self):
        return "{}()".format(
            self.__class__.__name__)
