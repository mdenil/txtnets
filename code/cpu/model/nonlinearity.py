__author__ = 'mdenil'

import numpy as np

from cpu.model import layer

import generic.model.nonlinearity

class Relu(generic.model.nonlinearity.Relu, layer.Layer):
    def _fprop(self, X):
        return np.maximum(0.0, X)

    def _bprop(self, delta, Y):
        return delta * (Y > 0.0)


class Tanh(generic.model.nonlinearity.Tanh, layer.Layer):
    def _fprop(self, X):
        return np.tanh(X)

    def _bprop(self, delta, Y):
        return delta * (1-Y**2)
