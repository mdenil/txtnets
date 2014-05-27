__author__ = 'mdenil'


import numpy as np
import generic.optimize.objective.contrastive_multilingual

__all__ = ["ContrastiveMultilingualEmbeddingObjective"]


class GaussianEnergy(object):
    def fprop(self, x, y):
        return 0.5 * np.sum((x - y)**2, axis=1, keepdims=True)

    def bprop(self, x, y, delta):
        delta_x = delta * (x - y)
        delta_y = delta * (y - x)
        return delta_x, delta_y


class ContrastiveHingeLoss(object):
    def __init__(self, margin):
        self.margin = margin

    def fprop(self, x_clean, x_noise):
        return np.mean(np.maximum(0.0, self.margin + x_clean - x_noise))

    def bprop(self, x_clean, x_noise, delta):
        delta *= (self.margin + x_clean - x_noise > 0)
        delta = delta.astype(np.float)
        delta /= float(x_clean.shape[0])

        return delta, -delta


class SquareExponentialLoss(object):
    def __init__(self, margin):
        self.margin = margin

    def fprop(self, e_clean, e_dirty):
        return np.mean(0.5 * e_clean**2 + self.margin * np.exp(-e_dirty))

    def bprop(self, e_clean, e_dirty, delta):
        delta /= float(e_clean.shape[0])
        d_clean = delta * e_clean
        d_dirty = - delta * self.margin * np.exp(-e_dirty)
        return d_clean, d_dirty


class SquareSquareMarginLoss(object):
    def __init__(self, margin):
        self.margin = margin

    def fprop(self, e_clean, e_dirty):
        return np.mean(0.5 * e_clean**2 + 0.5 * np.maximum(0, self.margin - e_dirty)**2)

    def bprop(self, e_clean, e_dirty, delta):
        delta /= float(e_clean.shape[0])
        d_clean = delta * e_clean
        d_dirty = -delta * np.maximum(0.0, self.margin - e_dirty)
        # d_dirty = -delta * (self.margin - e_dirty) * (e_dirty < self.margin)
        return d_clean, d_dirty


class ContrastiveMultilingualEmbeddingObjective(
        generic.optimize.objective.contrastive_multilingual.ContrastiveMultilingualEmbeddingObjective):

    Energy = GaussianEnergy
    # LossFunction = ContrastiveHingeLoss
    LossFunction = SquareSquareMarginLoss

    def _zeros_like(self, x):
        return np.zeros_like(x)
