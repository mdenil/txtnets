__author__ = 'mdenil'

import numpy as np

from cpu import space
from cpu.model import layer

import generic.model.pooling


# FIXME: move the CPU specific parts of KMaxPooling here
class KMaxPooling(generic.model.pooling.KMaxPooling, layer.Layer):
    pass


class SumFolding(generic.model.pooling.SumFolding, layer.Layer):
    def _fprop(self, X, X_space):
        X = np.sum(X, axis=X_space.axes.index('d2'))
        X_space = X_space.without_axes('d2')
        return X, X_space

    # bprop is entirely generic
    # no grads


class MaxFolding(generic.model.pooling.MaxFolding, layer.Layer):
    def _fprop(self, X):
        folded_size = X.shape[0] // 2

        switches = X[:folded_size] > X[folded_size:]
        switches = np.concatenate(
            [switches, np.logical_not(switches)],
            axis=0)

        Y = np.maximum(X[:folded_size], X[folded_size:])

        return Y, switches

    # bprop is entirely generic
    # no grads
