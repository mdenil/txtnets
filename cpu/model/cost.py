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
        back = Y - Y_true
        meta['space_below'] = fprop_state['input_space']
        return back, meta

    def __repr__(self):
        return "{}()".format(
            self.__class__.__name__)