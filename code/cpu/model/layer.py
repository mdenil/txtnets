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
        return np.concatenate([p.ravel() for p in self.params()])

    def unpack(self, values):
        start = 0
        for param in self.params():
            end = start + param.size
            # assign to values in the array referenced by param
            param.ravel()[:] = values[start:end]
            start = end

    def __repr__(self):
        return "{}()".format(
            self.__class__.__name__)
