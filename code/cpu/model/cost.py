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
        assert meta['space_below'].is_compatible_shape(delta)
        return delta, meta

    def __repr__(self):
        return "{}()".format(
            self.__class__.__name__)


class SquaredError(object):
    def fprop(self, Y, Y_true, meta):
        if not Y.shape == Y_true.shape:
            raise ValueError("Shape of predictions and labels do not match. (Y={}, Y_true={})".format(Y.shape, Y_true.shape))

        out = 0.5 * np.sum((Y - Y_true)**2, axis=1).mean()

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


# TODO: I think this is old and useless
class LargeMarginCost(object):
    def __init__(self, margin):
        self.margin = margin

    def fprop(self, Y_clean, Y_dirty, meta):

        unhinged_loss = self.margin - Y_clean + Y_dirty
        switches = unhinged_loss > 0.0
        hinged_loss = np.maximum(0.0, unhinged_loss)

        fprop_state = {
            'space_below': meta['space_below'],
            'switches': switches,
        }

        return hinged_loss.mean(), meta, fprop_state

    def bprop(self, Y_clean, Y_dirty, meta, fprop_state):
        meta['space_below'] = fprop_state['space_below']

        delta_clean = - Y_clean * fprop_state['switches']
        delta_dirty = Y_dirty * fprop_state['switches']

        delta_clean /= Y_clean.shape[0]
        delta_dirty /= Y_dirty.shape[0]

        return delta_clean, delta_dirty, meta

    def __repr__(self):
        return "{}(margin={})".format(
            self.__class__.__name__,
            self.margin)

