__author__ = 'mdenil'

import numpy as np

import cpu.space
import cpu.model.layer

import generic.model.encoding


class DictionaryEncoding(generic.model.encoding.DictionaryEncoding, cpu.model.layer.Layer):
    def _fprop(self, X):
        X = np.vstack([np.atleast_2d(x) for x in X])
        X_space = cpu.space.CPUSpace.infer(X, ('b', 'w'))

        return X, X_space