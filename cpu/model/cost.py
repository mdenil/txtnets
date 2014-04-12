__author__ = 'mdenil'

import numpy as np

class CrossEntropy(object):
    def __init__(self, stabilizer=1e-6):
        """
        :param stabilizer:  A small value used to prevent division by zero in bprop.  Will make the derivatives slightly inaccurate.
        :return:
        """
        self._stabilizer = stabilizer

    def fprop(self, Y, Y_true, meta):
        # really bad things happen if Y_true is a bool array

        if not Y.shape == Y_true.shape:
            raise ValueError("Shape of predictions and labels do not match. (Y={}, Y_true={})".format(Y.shape, Y_true.shape))

        out = - np.sum(Y_true * np.log(Y), axis=1).mean()

        fprop_state = {}
        fprop_state['input_space'] = meta['space_below']

        return out, meta, fprop_state


    def bprop(self, Y, Y_true, meta, fprop_state):

        if not Y.shape == Y_true.shape:
            raise ValueError("Shape of predictions and labels do not match. (Y={}, Y_true={})".format(Y.shape, Y_true.shape))

        delta = - Y_true / np.maximum(Y, self._stabilizer) + (1-Y_true) / np.maximum(1-Y, self._stabilizer)

        delta *= 1.0 / Y_true.shape[0]

        meta['space_below'] = fprop_state['input_space']
        assert meta['space_below'].is_compatable_shape(delta)
        return delta, meta

    def __repr__(self):
        return "{}()".format(
            self.__class__.__name__)