__author__ = 'mdenil'

import numpy as np
import scipy.optimize

import unittest
from cpu import model
from cpu import space

class KMaxPooling(unittest.TestCase):
    def setUp(self):
        w,f,d,b = 10, 2, 3, 5
        self.k = 4
        self.X = np.random.standard_normal(size=(d,f,b,w))
        self.X_space = space.Space.infer(self.X, ['d', 'f', 'b', 'w'])

        self.layer = model.pooling.KMaxPooling(k=self.k)
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



class SumFolding(unittest.TestCase):
    def setUp(self):
        w,f,d,b = 3, 1, 20, 3
        self.X = np.random.standard_normal(size=(d,f,b,w))
        self.X_space = space.Space.infer(self.X, ['d', 'f', 'b', 'w'])

        self.layer = model.pooling.SumFolding()
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
