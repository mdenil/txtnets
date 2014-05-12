__author__ = 'mdenil'

import numpy as np
import scipy.optimize

import unittest
from cpu import space

import cpu.model.pooling
import cpu.model.cost

model = cpu.model


class KMaxPooling(unittest.TestCase):
    def setUp(self):
        w,f,d,b = 10, 2, 3, 5
        self.k = 4
        self.X = np.random.standard_normal(size=(d,f,b,w))
        self.X_space = space.CPUSpace.infer(self.X, ['d', 'f', 'b', 'w'])

        self.layer = model.pooling.KMaxPooling(self.k)
        self.meta = {'space_below': self.X_space, 'lengths': np.random.randint(low=1, high=w, size=b)}

    def test_fprop(self):
        self.skipTest("WRITEME")

    def test_bprop(self):
        def func(x):
            x = x.reshape(self.X.shape)
            Y, meta, fprop_state = self.layer.fprop(x, meta=dict(self.meta))
            return Y.sum()

        def grad(x):
            X = x.reshape(self.X.shape)
            Y, meta, fprop_state = self.layer.fprop(X, meta=dict(self.meta))
            delta, meta = self.layer.bprop(np.ones_like(Y), meta=dict(meta), fprop_state=fprop_state)
            delta, _ = meta['space_below'].transform(delta, self.X_space.axes)
            return delta.ravel()

        assert scipy.optimize.check_grad(func, grad, self.X.ravel()) < 1e-5


class DynamicKMaxPooling(unittest.TestCase):
    def setUp(self):
        self.k = 4
        self.k_dynamic = 0.4

        self.X = np.array(
            [
                [5, 3, 1, 6, 2, 1],
                [2, 0, 1, 0, 8, 9],
                [1, 7, 1, 8, 2, 9],
            ]).astype(np.float)
        self.X_space = space.CPUSpace.infer(self.X, ['b', 'w'])
        self.lengths = np.array([4, 6, 2])

        self.layer = model.pooling.KMaxPooling(k=self.k, k_dynamic=self.k_dynamic)
        self.meta = {
            'space_below': self.X_space,
            'lengths': self.lengths,
        }

    def test_fprop(self):
        Y, meta, state = self.layer.fprop(self.X, meta=dict(self.meta))

        self.assertEqual(Y.shape[1], np.maximum(self.k, np.max(np.ceil(self.lengths * self.k_dynamic))))

        Y_expected = np.array([
            [5, 3, 1, 6],
            [2, 1, 8, 9],
            [1, 7, 0, 0],
        ])

        self.assertTrue(np.all(Y == Y_expected))


    def test_bprop(self):
        def func(x):
            x = x.reshape(self.X.shape)
            Y, meta, fprop_state = self.layer.fprop(x, meta=dict(self.meta))
            return Y.sum()

        def grad(x):
            X = x.reshape(self.X.shape)
            Y, meta, fprop_state = self.layer.fprop(X, meta=dict(self.meta))
            delta, meta = self.layer.bprop(np.ones_like(Y), meta=dict(meta), fprop_state=fprop_state)
            delta, _ = meta['space_below'].transform(delta, self.X_space.axes)

            return delta.ravel()

        assert scipy.optimize.check_grad(func, grad, self.X.ravel()) < 1e-5





class SumFolding(unittest.TestCase):
    def setUp(self):
        w,f,d,b = 3, 1, 20, 3
        self.X = np.random.standard_normal(size=(d,f,b,w))
        self.X_space = space.CPUSpace.infer(self.X, ['d', 'f', 'b', 'w'])

        self.layer = model.pooling.SumFolding()
        self.meta = {
            'space_below': self.X_space,
            'lengths': np.random.randint(low=1, high=w, size=b)
        }


    def test_fprop(self):
        self.skipTest("WRITEME")

    def test_bprop(self):
        def func(x):
            x = x.reshape(self.X.shape)
            Y, meta, fprop_state = self.layer.fprop(x, meta=dict(self.meta))
            return Y.sum()

        def grad(x):
            X = x.reshape(self.X.shape)
            Y, meta, fprop_state = self.layer.fprop(X, meta=dict(self.meta))
            delta, meta = self.layer.bprop(np.ones_like(Y), meta=dict(meta), fprop_state=fprop_state)
            delta, _ = meta['space_below'].transform(delta, self.X_space.axes)
            return delta.ravel()

        assert scipy.optimize.check_grad(func, grad, self.X.ravel()) < 1e-5



class MaxFolding(unittest.TestCase):
    def setUp(self):
        w,f,d,b = 3, 1, 20, 3
        self.X = np.random.standard_normal(size=(d,f,b,w))
        self.X_space = space.CPUSpace.infer(self.X, ['d', 'f', 'b', 'w'])

        self.layer = model.pooling.MaxFolding()
        self.meta = {'space_below': self.X_space, 'lengths': np.random.randint(low=1, high=w, size=b)}


    def test_fprop(self):
        self.skipTest("WRITEME")

    def test_bprop(self):
        def func(x):
            x = x.reshape(self.X.shape)
            Y, meta, fprop_state = self.layer.fprop(x, meta=dict(self.meta))
            return Y.sum()

        def grad(x):
            X = x.reshape(self.X.shape)
            Y, meta, fprop_state = self.layer.fprop(X, meta=dict(self.meta))
            delta, meta = self.layer.bprop(np.ones_like(Y), meta=dict(meta), fprop_state=fprop_state)
            delta, _ = meta['space_below'].transform(delta, self.X_space.axes)
            return delta.ravel()

        assert scipy.optimize.check_grad(func, grad, self.X.ravel()) < 1e-5
