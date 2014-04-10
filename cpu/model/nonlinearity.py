__author__ = 'mdenil'

import numpy as np

from cpu.model import layer

class Relu(layer.Layer):
    def __init__(self):
        pass

    def fprop(self, X, meta):
        Y = np.maximum(0, X)

        fprop_state = {
            "input_space": meta['space_below'],
            "Y": Y,
        }

        return Y, meta, fprop_state

    def bprop(self, delta, meta, fprop_state):
        Y = fprop_state['Y']

        delta, meta['space_below'] = meta['space_above'].transform(delta, fprop_state['input_space'].axes)

        back = delta * (Y > 0)
        return back, meta

    def __repr__(self):
        return "{}()".format(
            self.__class__.__name__)



class Tanh(layer.Layer):
    def __init__(self):
        pass

    def fprop(self, X, meta):
        Y = np.tanh(X)

        fprop_state = {
            "input_space": meta['space_below'],
            "Y": Y,
        }

        meta['space_above'] = meta['space_below']

        return Y, meta, fprop_state

    def bprop(self, delta, meta, fprop_state):
        Y = fprop_state['Y']

        delta, meta['space_below'] = meta['space_above'].transform(delta, fprop_state['input_space'].axes)

        back = delta * (1-Y**2)
        return back, meta

    def __repr__(self):
        return "{}()".format(
            self.__class__.__name__)
