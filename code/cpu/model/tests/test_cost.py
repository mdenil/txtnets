__author__ = 'mdenil'

import numpy as np
import scipy.optimize

import unittest
from cpu import model
from cpu import space

class CrossEntropy(unittest.TestCase):
    def setUp(self):
        self.n_classes = 5
        self.n_data = 2

        self.cost = model.cost.CrossEntropy(stabilizer=0.0)

        self.Y = np.random.uniform(size=(self.n_data, self.n_classes))
        self.Y /= self.Y.sum(axis=1, keepdims=True)
        self.meta = {
            'space_below': space.CPUSpace.infer(self.Y, ['b', 'c'])
        }

        self.Y_true = np.random.randint(0, self.n_classes-1, size=self.n_data)
        self.Y_true = np.equal.outer(self.Y_true, np.arange(self.n_classes)).astype(self.Y.dtype)
        # self.Y_true = np.random.uniform(size=(self.n_data, self.n_classes))
        # self.Y_true /= self.Y_true.sum(axis=1, keepdims=True)


    def test_fprop(self):
        actual, _, _ = self.cost.fprop(self.Y, self.Y_true, meta=self.meta)
        expected = -np.sum(self.Y_true * np.log(self.Y)) / self.n_data

        assert np.allclose(actual, expected)

    def test_bprop(self):
        def func(y):
            y = y.reshape((self.n_data, self.n_classes - 1))
            y = np.hstack([y, 1-np.sum(y, axis=1, keepdims=True)])
            assert np.allclose(np.sum(y, axis=1),  1)
            assert np.all(y > 0)
            assert np.allclose(np.sum(self.Y_true, axis=1), 1)
            c, meta, cost_state = self.cost.fprop(y, self.Y_true, meta=dict(self.meta))
            return c

        def grad(y):
            y = y.reshape((self.n_data, self.n_classes - 1))
            y = np.hstack([y, 1-np.sum(y, axis=1, keepdims=True)])
            assert np.allclose(np.sum(y, axis=1),  1)
            assert np.all(y > 0)
            assert np.allclose(np.sum(self.Y_true, axis=1), 1)
            cost, meta, cost_state = self.cost.fprop(y, self.Y_true, meta=dict(self.meta))
            delta, meta = self.cost.bprop(y, self.Y_true, meta=dict(meta), fprop_state=cost_state)
            delta = delta[:,:-1]
            return delta.ravel()

        Y = self.Y[:,:-1]
        #
        # print
        # print func(Y.ravel().copy())
        # print grad(Y.ravel().copy()).reshape(Y.shape)
        # print scipy.optimize.approx_fprime(Y.ravel().copy(), func, 1e-8).reshape(Y.shape)
        #
        # print scipy.optimize.check_grad(func, grad, Y.ravel().copy())
        #
        # assert scipy.optimize.check_grad(func, grad, Y.ravel()) < 1e-5

        self.skipTest('FIXME')