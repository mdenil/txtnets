__author__ = 'mdenil'

import numpy as np
import scipy.optimize

import unittest
from cpu import model
from cpu import space

class WordEmbedding(unittest.TestCase):
    def setUp(self):
        d = 10
        vocabulary_size = 10

        self.layer = model.embedding.WordEmbedding(
            dimension=d,
            vocabulary_size=vocabulary_size)

        self.words = np.random.randint(vocabulary_size, size=(3,5))

        self.words_space = space.Space.infer(self.words, ['b', 'w'])
        self.meta = {'lengths': np.zeros_like(self.words) + self.words.shape[1], 'space_below': self.words_space}


    def test_fprop(self):
        actual, meta, fprop_state = self.layer.fprop(self.words, meta=self.meta)
        expected = self.layer.E[self.words.ravel()]
        assert np.allclose(actual, expected)

    def test_bprop(self):
        self.skipTest('WRITEME')

    def test_grad(self):
        def func(E):
            self.layer.E = E.reshape(self.layer.E.shape)
            Y, _, _ = self.layer.fprop(self.words.copy(), meta=self.meta)
            c = Y.sum()
            return Y.sum()

        def grad(E):
            self.layer.E = E.reshape(self.layer.E.shape)

            Y, meta, fprop_state = self.layer.fprop(self.words.copy(), meta=self.meta)
            delta = np.ones_like(Y)
            [grad_E] = self.layer.grads(delta, meta=meta, fprop_state=fprop_state)

            return grad_E.ravel()

        assert scipy.optimize.check_grad(func, grad, self.layer.E.ravel()) < 1e-5
