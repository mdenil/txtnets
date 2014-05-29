__author__ = 'mdenil, albandemiraj'

import numpy as np

import cpu.space
import cpu.model.layer

import generic.model.encoding


class DictionaryEncoding(generic.model.encoding.DictionaryEncoding, cpu.model.layer.Layer):
    def _fprop(self, X, meta):
        X = np.vstack([np.atleast_2d(x) for x in X])
        X_space = meta["space_below"]

        return X, X_space