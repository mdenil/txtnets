__author__ = 'mdenil'

import numpy as np

from collections import OrderedDict

from cpu import space
from cpu.model import layer
import generic.model.embedding


class WordEmbedding(generic.model.embedding.WordEmbedding, layer.Layer):

    def _fprop(self, X):
        return self.E[X]

    def _bprop(self, delta, Y):
        return np.sum(Y * delta, axis=1)

    def _grads(self, delta, X):
        grad_E = np.zeros_like(self.E)
        for i,j in enumerate(X.ravel()):
            grad_E[j] += delta[i]

        grad_E[self.padding] = 0.0

        return [grad_E]

    # def indicator_matrix(self, X, meta):
    #     X, working_space = meta['space_below'].transform(X, [('b','w')])
    #
    #     I = np.zeros((X.size, self.vocabulary_size))
    #     I[np.arange(X.size), X.ravel()] = 1
    #
    #     I_extent = OrderedDict()
    #     I_extent['b'] = working_space.get_extent('b')
    #     I_extent['w'] = working_space.get_extent('w')
    #     I_extent['d'] = self.vocabulary_size
    #     I_space = space.CPUSpace([('b', 'w'), 'd'], I_extent)
    #
    #     return I, I_space
