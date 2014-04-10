__author__ = 'mdenil'

import numpy as np

from collections import OrderedDict

from cpu import space
from cpu.model import layer

class WordEmbedding(layer.Layer):
    def __init__(self,
                 dimension,
                 vocabulary_size,
                 ):

        self.dimension = dimension
        self.vocabulary_size = vocabulary_size

        self.E = 0.05 * np.random.standard_normal(size=(self.vocabulary_size, self.dimension))

    def fprop(self, X, meta):
        working_space = meta['space_below']

        X, working_space = working_space.transform(X, ['bw', 'd'])

        Y = self.E[X.ravel()]

        meta['space_above'] = working_space.with_extent(d=self.dimension)
        fprop_state = {
            'input_space': meta['space_below'],
            'embedding_space': meta['space_above'],
            'X': X,
            'Y': Y,
        }

        return Y, meta, fprop_state

    def bprop(self, delta, meta, fprop_state):
        Y = fprop_state['Y']

        delta_space = meta['space_above']

        delta, delta_space = delta_space.transform(delta, fprop_state['embedding_space'].axes)

        delta = np.sum(Y * delta, axis=1)
        delta_space = delta_space.without_axes('d')
        delta, delta_space = delta_space.transform(delta, ['b', 'w'])

        meta['space_below'] = delta_space

        return delta, meta

    def grads(self, delta, meta, fprop_state):
        delta_space = meta['space_above']
        X = fprop_state['X']
        X_space = fprop_state['input_space']

        delta, delta_space = delta_space.transform(delta, ['bw', 'd'])
        X, X_space = X_space.transform(X, ['bw', 'd'])

        grad_E = np.zeros_like(self.E)
        for i in X.ravel():
            grad_E[i] += delta[i]

        return [grad_E]

    def params(self):
        return [self.E]

    def __repr__(self):
        return "{}(dim={}, vocab_size={})".format(
            self.__class__.__name__,
            self.dimension,
            self.vocabulary_size)

    def indicator_matrix(self, X, meta):
        X, working_space = meta['space_below'].transform(X, ['bw'])

        I = np.zeros((X.size, self.vocabulary_size))
        I[np.arange(X.size), X.ravel()] = 1

        I_extent = OrderedDict()
        I_extent['b'] = working_space.get_extent('b')
        I_extent['w'] = working_space.get_extent('w')
        I_extent['d'] = self.vocabulary_size
        I_space = space.Space(['bw', 'd'], I_extent)

        return I, I_space
