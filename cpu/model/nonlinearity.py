__author__ = 'mdenil'

import numpy as np

class Relu(object):
    def __init__(self):
        self.input_axes = ['b', 'w', 'f', 'd']
        self.output_axes = ['b', 'w', 'f', 'd']

    def fprop(self, X, **meta):
        X = np.maximum(0, X)
        return X, meta

    def __repr__(self):
        return "{}()".format(
            self.__class__.__name__)



class Tanh(object):
    def __init__(self):
        self.input_axes = ['b', 'w', 'f', 'd']
        self.output_axes = ['b', 'w', 'f', 'd']

    def fprop(self, X, **meta):
        X = np.tanh(X)
        return X, meta

    def __repr__(self):
        return "{}()".format(
            self.__class__.__name__)
