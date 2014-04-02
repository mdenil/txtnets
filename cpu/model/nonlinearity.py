__author__ = 'mdenil'

import numpy as np

class Relu(object):
    def __init__(self):
        self.input_axes = ['b', 'w', 'f', 'd']
        self.output_axes = ['b', 'w', 'f', 'd']

    def fprop(self, X, **meta):
        self.Y = np.maximum(0, X)
        return self.Y, meta

    # def bprop(self, delta, **meta):
    #     back = delta * (self.Y > 0)
    #     return back, meta

    def __repr__(self):
        return "{}()".format(
            self.__class__.__name__)



class Tanh(object):
    def __init__(self):
        self.input_axes = ['b', 'w', 'f', 'd']
        self.output_axes = ['b', 'w', 'f', 'd']

    def fprop(self, X, **meta):
        self.Y = np.tanh(X)
        return self.Y, meta

    # def bprop(self, delta, meta):
    #     back = delta * (1-self.Y**2)
    #     return back, meta

    def __repr__(self):
        return "{}()".format(
            self.__class__.__name__)
