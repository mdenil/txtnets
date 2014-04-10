__author__ = 'mdenil'

import numpy as np
import scipy.optimize

import unittest
from cpu import model
from cpu import space

class CrossEntropy(unittest.TestCase):
    def setUp(self):
        self.n_classes = 5
        self.n_data = 50

        self.cost = model.cost.CrossEntropy()

        self.Y = np.random.uniform(size=(self.n_data, self.n_classes))
        self.Y /= self.Y.sum(axis=1).reshape((-1,1))
        self.meta = {
            'space_below': space.Space.infer(self.Y, ['b', 'c'])
        }

        self.Y_true = np.random.randint(0, self.n_classes, size=self.n_data)
        self.Y_true = np.equal.outer(self.Y_true, np.arange(self.n_classes)).astype(self.Y.dtype)

    def test_fprop(self):
        actual, _, _ = self.cost.fprop(self.Y, self.Y_true, meta=self.meta)
        expected = -np.sum(self.Y_true * np.log(self.Y))

        assert np.allclose(actual, expected)

    # def test_bprop(self):
    #     pass