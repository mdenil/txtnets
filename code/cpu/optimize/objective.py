__author__ = 'mdenil'

import numpy as np

import generic.model.utils
import generic.optimize.objective

from generic.optimize.objective import CostMinimizationObjective


###############


class _GaussianEnergy(object):
    def fprop(self, x, y):
        return 0.5 * np.sum((x - y)**2, axis=1, keepdims=True)

    def bprop(self, x, y, delta):
        delta_x = delta * x
        delta_y = delta * y
        return delta_x, delta_y


class _ContrastiveHingeLoss(object):
    def __init__(self, margin):
        self.margin = margin

    def fprop(self, x_clean, x_noise):
        return np.maximum(0.0, self.margin + x_clean - x_noise)

    def bprop(self, x_clean, x_noise):
        # TODO: avoid extra work here
        delta = self.fprop(x_clean, x_noise) > 0
        delta = delta.astype(np.float)
        delta /= float(x_clean.shape[0])
        delta_clean = delta * x_clean
        delta_dirty = delta * -x_noise
        return delta_clean, delta_dirty


class ContrastiveMultilingualEmbeddingObjective(
        generic.optimize.objective.ContrastiveMultilingualEmbeddingObjective):

    Energy = _GaussianEnergy
    LossFunction = _ContrastiveHingeLoss


    def _zeros_like(self, x):
        return np.zeros_like(x)

    def _mean(self, x):
        return np.mean(x)

