__author__ = 'mdenil'

import numpy as np

class WordEmbedding(object):
    def __init__(self,
                 dimension,
                 vocabulary_size,
                 E=None):
        self.dimension = dimension
        self.vocabulary_size = vocabulary_size

        if E is None:
            self.E = 0.0025 * np.random.standard_normal(size=(self.vocabulary_size, self.dimension))
        else:
            assert E.shape == (vocabulary_size, dimension)
            self.E = E

    def fprop(self, X, meta):
        X, X_space = meta['space_below'].transform(X, (('b', 'w'), 'd'))

        Y = self._fprop(X.ravel())
        Y_space = X_space.with_extents(d=self.dimension)

        fprop_state = {
            'X_space': X_space,
            'Y_space': Y_space,
            'X': X,
            'Y': Y,
        }

        # This is the standard order for axes in most other layers, we transform here instead of later so that the
        # transformation happens before any broadcasts.
        Y, Y_space = Y_space.transform(Y, (('d', 'b'), 'w'))
        meta['space_above'] = Y_space

        return Y, meta, fprop_state

    def bprop(self, delta, meta, fprop_state):
        Y = fprop_state['Y']
        Y_space = fprop_state['Y_space']

        delta_space = meta['space_above']

        delta, delta_space = delta_space.transform(delta, (('b', 'w'), 'd'))
        Y, Y_space = Y_space.transform(delta, (('b', 'w'), 'd'))

        delta = self._bprop(delta, Y)

        delta_space = delta_space.without_axes('d')

        meta['space_below'] = delta_space

        return delta, meta

    def grads(self, delta, meta, fprop_state):
        delta_space = meta['space_above']
        X = fprop_state['X']
        X_space = fprop_state['X_space']

        delta, delta_space = delta_space.transform(delta, (('b', 'w'), 'd'))
        X, X_space = X_space.transform(X, (('b', 'w'), 'd'))

        return self._grads(delta, X)

    def params(self):
        return [self.E]

    def __repr__(self):
        return "{}(dim={}, vocab_size={})".format(
            self.__class__.__name__,
            self.dimension,
            self.vocabulary_size)
