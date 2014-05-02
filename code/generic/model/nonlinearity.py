__author__ = 'mdenil'

import numpy as np

class Relu(object):
    def fprop(self, X, meta):
        Y = self._fprop(X)

        fprop_state = {
            "input_space": meta['space_below'],
            "Y": Y,
        }

        meta['space_above'] = meta['space_below']

        return Y, meta, fprop_state

    def bprop(self, delta, meta, fprop_state):
        Y = fprop_state['Y']

        delta, meta['space_below'] = meta['space_above'].transform(delta, fprop_state['input_space'].axes)

        back = self._bprop(delta, Y)

        return back, meta

    def __repr__(self):
        return "{}()".format(
            self.__class__.__name__)


class Tanh(object):
    def fprop(self, X, meta):
        Y = self._fprop(X)

        fprop_state = {
            "input_space": meta['space_below'],
            "Y": Y,
        }

        meta['space_above'] = meta['space_below']

        return Y, meta, fprop_state

    def bprop(self, delta, meta, fprop_state):
        Y = fprop_state['Y']

        delta, meta['space_below'] = meta['space_above'].transform(delta, fprop_state['input_space'].axes)

        back = self._bprop(delta, Y)
        return back, meta

    def __repr__(self):
        return "{}()".format(
            self.__class__.__name__)