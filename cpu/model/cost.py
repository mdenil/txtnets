__author__ = 'mdenil'

import numpy as np

class CrossEntropy(object):
    def __init__(self):
        pass

    def fprop(self, Y, Y_true, meta):
        # really bad things happen if Y_true is a bool array

        out = - np.sum(Y_true * np.log(Y))

        fprop_state = {}
        fprop_state['input_space'] = meta['space_below']

        return out, meta, fprop_state


    def bprop(self, Y, Y_true, meta, fprop_state):
        # delta = Y - Y_true
        delta = - Y_true / Y + (1-Y_true) / (1-Y)
        meta['space_below'] = fprop_state['input_space']
        return delta, meta

    def __repr__(self):
        return "{}()".format(
            self.__class__.__name__)