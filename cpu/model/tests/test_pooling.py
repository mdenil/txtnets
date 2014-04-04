__author__ = 'mdenil'

import numpy as np
import scipy.optimize

import unittest
from cpu import model
from cpu import space

# FIXME: test_bprop is identical for many different tests... can probably be shared.

class KMaxPooling(unittest.TestCase):
    def setUp(self):
        w,f,d,b = 6, 1, 1, 5
        self.k = 4
        self.n_classes = 7
        self.X = np.random.standard_normal(size=(d,f,b,w))
        self.X_space = space.Space.infer(self.X, ['d', 'f', 'b', 'w'])
        self.Y = np.random.randint(0, self.n_classes, size=b)
        self.Y = np.equal.outer(np.arange(self.n_classes), self.Y).astype(self.X.dtype)

        self.layer = model.pooling.KMaxPooling(k=self.k)
        self.meta = {'data_space': self.X_space, 'lengths': np.random.randint(low=1, high=w, size=b)}

        self.csm = model.model.CSM(
            input_axes=['d', 'f', 'b', 'w'],
            layers=[
                self.layer,
                model.transfer.Softmax(
                    n_classes=self.n_classes,
                    n_input_dimensions=self.k*f*d),
                ])
        self.cost = model.cost.CrossEntropy()

    def test_fprop(self):
        self.skipTest("WRITEME")

    def test_bprop(self):
        def func(x):
            x = x.reshape(self.X.shape)
            Y = self.csm.fprop(x, **self.meta)
            c,_ = self.cost.fprop(Y, self.Y)
            return c

        def grad(x):
            X = x.reshape(self.X.shape)
            Y = self.csm.fprop(X, **self.meta)
            delta, _ = self.cost.bprop(Y, self.Y)
            delta = self.csm.bprop(delta)
            return delta.ravel()

        assert scipy.optimize.check_grad(func, grad, self.X.ravel()) < 1e-5



class SumFolding(unittest.TestCase):
    def setUp(self):
        w,f,d,b = 3, 1, 20, 3
        self.n_classes = 7
        self.X = np.random.standard_normal(size=(d,f,b,w))
        self.X_space = space.Space.infer(self.X, ['d', 'f', 'b', 'w'])
        self.Y = np.random.randint(0, self.n_classes, size=b)
        self.Y = np.equal.outer(np.arange(self.n_classes), self.Y).astype(self.X.dtype)

        self.layer = model.pooling.SumFolding()
        self.meta = {'data_space': self.X_space, 'lengths': np.random.randint(low=1, high=w, size=b)}

        self.csm = model.model.CSM(
            input_axes=['d', 'f', 'b', 'w'],
            layers=[
                self.layer,
                model.transfer.Softmax(
                    n_classes=self.n_classes,
                    n_input_dimensions=w*f*d / 2),
                ])
        self.cost = model.cost.CrossEntropy()

    def test_fprop(self):
        self.skipTest("WRITEME")

    def test_bprop(self):
        def func(x):
            x = x.reshape(self.X.shape)
            Y = self.csm.fprop(x, **self.meta)
            c,_ = self.cost.fprop(Y, self.Y)
            return c

        def grad(x):
            X = x.reshape(self.X.shape)
            Y = self.csm.fprop(X, **self.meta)
            delta, _ = self.cost.bprop(Y, self.Y)
            delta = self.csm.bprop(delta)
            return delta.ravel()

        assert scipy.optimize.check_grad(func, grad, self.X.ravel()) < 1e-5
