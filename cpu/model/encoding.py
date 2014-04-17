__author__ = 'mdenil'

import numpy as np

from cpu import space
from cpu.model import layer

class DictionaryEncoding(layer.Layer):
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def fprop(self, X, meta):
        X = [self._encode(x) for x in X]
        X = np.vstack([np.atleast_2d(x) for x in X])

        X_space = space.Space.infer(X, ['b', 'w'])

        meta = {
            'lengths': meta['lengths'],
            'space_above': X_space,
            }

        fprop_state = {}

        return X, meta, fprop_state

    def _encode(self, x):
        return [self.vocabulary[c] for c in x]

    def __repr__(self):
        return "{}(vocabulary_size={})".format(
            self.__class__.__name__,
            len(self.vocabulary))