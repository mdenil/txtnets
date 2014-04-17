__author__ = 'mdenil'

import numpy as np

class L2Regularizer(object):
    def __init__(self, lamb):
        self.lamb = lamb

    def regularize(self, model):
        params = model.params()

        cost = sum(self.lamb * np.sum(p**2) for p in params)
        grads = [2.0 * self.lamb * p for p in params]

        return cost, grads