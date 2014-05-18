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
