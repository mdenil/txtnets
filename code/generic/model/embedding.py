__author__ = 'mdenil'

import numpy as np

class WordEmbedding(object):
    def __init__(self,
                 dimension,
                 vocabulary_size,
                 padding,
                 E=None):
        self.dimension = dimension
        self.padding = padding
        self.vocabulary_size = vocabulary_size

        if E is None:
            self.E = 0.01 * np.random.standard_normal(size=(self.dimension, self.vocabulary_size))

            self.E[:, padding] = 0.0
        else:
            assert E.shape == (dimension, vocabulary_size)
            self.E = E

    def fprop(self, X, meta):
        X, X_space = meta['space_below'].transform(X, ('b', 'd', 'w'))

        Y = self._fprop(X)
        Y_space = X_space.with_extents(d=self.dimension)

        fprop_state = {
            'X_space': X_space,
            'Y_space': Y_space,
            'X': X,
            'Y': Y,
        }

        # This is the standard order for axes in most other layers, we transform here instead of later so that the
        # transformation happens before any broadcasts.
        Y, Y_space = Y_space.transform(Y, (('b', 'd'), 'w'))
        meta['space_above'] = Y_space

        return Y, meta, fprop_state

    def bprop(self, delta, meta, fprop_state):
        Y = fprop_state['Y']
        Y_space = fprop_state['Y_space']

        delta_space = meta['space_above']

        delta, delta_space = delta_space.transform(delta, ('b', 'd', 'w'))
        Y, Y_space = Y_space.transform(Y, ('b', 'd', 'w'))

        delta = self._bprop(delta, Y, delta_space)

        delta_space = delta_space.without_axes('d')

        meta['space_below'] = delta_space

        return delta, meta

    def grads(self, delta, meta, fprop_state):
        delta_space = meta['space_above']
        X = fprop_state['X']
        X_space = fprop_state['X_space']

        delta, delta_space = delta_space.transform(delta, ('b', 'd', 'w'))
        X, X_space = X_space.transform(X, ('b', 'd', 'w'))

        return self._grads(delta, X)

    def params(self):
        return [self.E]

    def __repr__(self):
        return "{}(dim={}, vocab_size={})".format(
            self.__class__.__name__,
            self.dimension,
            self.vocabulary_size)
