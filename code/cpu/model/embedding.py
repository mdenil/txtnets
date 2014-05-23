__author__ = 'mdenil'

import numpy as np

from collections import OrderedDict

from cpu import space
from cpu.model import layer
import generic.model.embedding


class WordEmbedding(generic.model.embedding.WordEmbedding, layer.Layer):

    def _fprop(self, X):
        return self.E[:, X.squeeze()].transpose((1, 0, 2))

    def _bprop(self, delta, Y, space):
        return np.sum(Y * delta, axis=space.axes.index('d'))

    def _grads(self, delta, X):
        # delta is b,d,w
        # X is b,d=1,w
        # E is d, w

        grad_E = np.zeros_like(self.E)
        delta = delta.transpose((0, 2, 1)) # b w d
        delta = delta.reshape((-1, delta.shape[-1])) # (b, w), d
        for i, j in enumerate(X.ravel()):
            grad_E[:, j] += delta[i]

        grad_E[:, self.padding] = 0.0

        return [grad_E]
