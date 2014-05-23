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


class ContrastiveMultilingualEmbeddingObjective(
        generic.optimize.objective.contrastive_multilingual.ContrastiveMultilingualEmbeddingObjective):

    Energy = GaussianEnergy
    LossFunction = ContrastiveHingeLoss

    def _zeros_like(self, x):
        return np.zeros_like(x)
