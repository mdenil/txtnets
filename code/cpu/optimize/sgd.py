__author__ = 'mdenil'

import numpy as np

import generic.optimize.sgd

class SGD(generic.optimize.sgd.SGD):
    def _mean_abs(self, x):
        return np.mean(np.abs(x))
