__author__ = 'mdenil'

import numpy as np
import scipy.optimize

import unittest
from cpu import model
from cpu import space

class Tanh(unittest.TestCase):
    def setUp(self):
        w,f,d,b = 2, 3, 5, 10
        self.n_classes = 7
        self.X = np.random.standard_normal(size=(w, f, d, b))
        self.X_space = space.Space.infer(self.X, ['w', 'f', 'd', 'b'])
        self.Y = np.random.randint(0, self.n_classes, size=b)
        self.Y = np.equal.outer(np.arange(self.n_classes), self.Y).astype(self.X.dtype)

        self.layer = model.nonlinearity.Tanh()
        self.meta = {'space_below': self.X_space, 'lengths': np.zeros(b) + w}

        self.csm = model.model.CSM(
            input_axes=['w', 'f', 'd', 'b'],
            layers=[
                self.layer,
                model.transfer.Softmax(
                    n_classes=self.n_classes,
                    n_input_dimensions=w*f*d),
                ])
        self.cost = model.cost.CrossEntropy()

    def test_fprop(self):
        actual, _ = self.layer.fprop(self.X, meta=self.meta)
        expected = np.tanh(self.X)

        assert np.allclose(actual, expected)

    def test_bprop(self):
        def func(x):
            x = x.reshape(self.X.shape)
            Y = self.csm.fprop(x, meta=self.meta)
            c,_ = self.cost.fprop(Y, self.Y)
            return c

        def grad(x):
            X = x.reshape(self.X.shape)
            Y, meta, fprop_state = self.csm.fprop(X, meta=self.meta, return_meta=True, return_state=True)
            delta, _ = self.cost.bprop(Y, self.Y)
            delta = self.csm.bprop(delta, fprop_state=fprop_state)
            return delta.ravel()

        assert scipy.optimize.check_grad(func, grad, self.X.ravel()) < 1e-5


class Relu(unittest.TestCase):
    def setUp(self):
        w,f,d,b = 2, 3, 5, 10
        self.n_classes = 7
        self.X = np.random.standard_normal(size=(w, f, d, b))
        self.X_space = space.Space.infer(self.X, ['w', 'f', 'd', 'b'])
        self.Y = np.random.randint(0, self.n_classes, size=b)
        self.Y = np.equal.outer(np.arange(self.n_classes), self.Y).astype(self.X.dtype)

        self.layer = model.nonlinearity.Relu()
        self.meta = {'space_below': self.X_space, 'lengths': np.zeros(b) + w}

        self.csm = model.model.CSM(
            input_axes=['w', 'f', 'd', 'b'],
            layers=[
                self.layer,
                model.transfer.Softmax(
                    n_classes=self.n_classes,
                    n_input_dimensions=w*f*d),
                ])
        self.cost = model.cost.CrossEntropy()

    def test_fprop(self):
        actual, _ = self.layer.fprop(self.X, meta=self.meta)
        expected = np.maximum(0, self.X)

        assert np.allclose(actual, expected)

    def test_bprop(self):
        def func(x):
            x = x.reshape(self.X.shape)
            Y = self.csm.fprop(x, meta=self.meta)
            c,_ = self.cost.fprop(Y, self.Y)
            return c

        def grad(x):
            X = x.reshape(self.X.shape)
            Y, meta, fprop_state = self.csm.fprop(X, meta=self.meta, return_meta=True, return_state=True)
            delta, _ = self.cost.bprop(Y, self.Y)
            delta = self.csm.bprop(delta, fprop_state=fprop_state)
            return delta.ravel()

        assert scipy.optimize.check_grad(func, grad, self.X.ravel()) < 1e-5

