__author__ = 'mdenil'

import pycuda.autoinit
from pycuda import cumath

import gpu.utils
from gpu import space
from gpu.model import layer

import scikits.cuda.linalg
scikits.cuda.linalg.init()


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

        out = - pycuda.gpuarray.sum(Y_true * cumath.log(Y)) / meta['space_below'].get_extent('b')

        fprop_state = {}
        fprop_state['input_space'] = meta['space_below']

        return out, meta, fprop_state

    def bprop(self, Y, Y_true, meta, fprop_state):

        if not Y.shape == Y_true.shape:
            raise ValueError("Shape of predictions and labels do not match. (Y={}, Y_true={})".format(Y.shape, Y_true.shape))

        delta = - Y_true / (Y + self._stabilizer) + (1.0 - Y_true) / (1.0 - Y + self._stabilizer)

        delta *= 1.0 / Y_true.shape[0]

        meta['space_below'] = fprop_state['input_space']
        assert meta['space_below'].is_compatible_shape(delta)
        return delta, meta

    def __repr__(self):
        return "{}()".format(
            self.__class__.__name__)


class SquaredError(object):
    def fprop(self, Y, Y_true, meta):
        if not Y.shape == Y_true.shape:
            raise ValueError("Shape of predictions and labels do not match. (Y={}, Y_true={})".format(Y.shape, Y_true.shape))

        out = 0.5 * pycuda.gpuarray.sum((Y - Y_true)**2) / meta['space_below'].get_extent('b')

        fprop_state = {}
        fprop_state['input_space'] = meta['space_below']

        return out, meta, fprop_state

    def bprop(self, Y, Y_true, meta, fprop_state):
        if not Y.shape == Y_true.shape:
            raise ValueError("Shape of predictions and labels do not match. (Y={}, Y_true={})".format(Y.shape, Y_true.shape))

        delta = Y - Y_true
        delta /= Y_true.shape[0]

        meta['space_below'] = fprop_state['input_space']
        assert meta['space_below'].is_compatible_shape(delta)
        return delta, meta
