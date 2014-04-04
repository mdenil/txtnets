__author__ = 'mdenil'

import numpy as np

class Relu(object):
    def __init__(self):
        pass

    def fprop(self, X, **meta):
        Y = np.maximum(0, X)
        return Y, meta

    def bprop(self, Y, delta, **meta):
        back = delta * (Y > 0)
        return back, meta

    def __repr__(self):
        return "{}()".format(
            self.__class__.__name__)



class Tanh(object):
    def __init__(self):
        pass

    def fprop(self, X, **meta):
        X = np.tanh(X)
        return X, meta

    def bprop(self, Y, delta, **meta):
        back = delta * (1-Y**2)
        return back, meta

    def __repr__(self):
        return "{}()".format(
            self.__class__.__name__)
