__author__ = 'mdenil'

import numpy as np
import scipy.optimize

import unittest
from cpu import model

class Softmax(unittest.TestCase):
    def setUp(self):
        # X = ['w', 'f', 'd', 'b']
        # Y = ['d', 'b'] (d = classes)
        w,f,d,b = 2, 3, 5, 10
        self.n_input_dimensions = w*f*d
        self.n_classes = 7

        self.model = model.transfer.Softmax(
            n_classes=self.n_classes,
            n_input_dimensions=self.n_input_dimensions)

        self.X = np.random.standard_normal(size=(w, f, d, b))
        self.Y = np.random.randint(0, self.n_classes, size=b)
        self.Y = np.equal.outer(np.arange(self.n_classes), self.Y).astype(self.X.dtype)

        self.meta = {'lengths': np.zeros(b) + w}

    def test_fprop(self):
        actual, _ = self.model.fprop(self.X, **self.meta)
        expected = np.exp(np.dot(self.model.W, self.X.reshape((self.n_input_dimensions, -1))) + self.model.b)
        expected /= np.sum(expected, axis=0)

        assert np.allclose(actual, expected)

    def test_bprop(self):
        cost = model.cost.CrossEntropy()

        def func(x):
            x = x.reshape(self.X.shape)
            Y, _ = self.model.fprop(x, **self.meta)
            c,_ = cost.fprop(Y, self.Y)
            return c

        def grad(x):
            X = x.reshape(self.X.shape)
            Y, _ = self.model.fprop(X, **self.meta)
            delta, _ = cost.bprop(Y, self.Y)
            delta, _ = self.model.bprop(delta)
            return delta.ravel()

        assert scipy.optimize.check_grad(func, grad, self.X.ravel()) < 1e-5

    def test_grad_W(self):
        cost = model.cost.CrossEntropy()

        def func(w):
            self.model.W = w.reshape(self.model.W.shape)
            Y, _ = self.model.fprop(self.X, **self.meta)
            c,_ = cost.fprop(Y, self.Y)
            return c

        def grad(w):
            self.model.W = w.reshape(self.model.W.shape)
            Y, _ = self.model.fprop(self.X, **self.meta)
            delta, _ = cost.bprop(Y, self.Y)
            [gw, _], _ = self.model.grads(self.X, delta, **self.meta)

            return gw.ravel()

        assert scipy.optimize.check_grad(func, grad, self.model.W.ravel()) < 1e-5

    def test_grad_b(self):
        cost = model.cost.CrossEntropy()

        def func(b):
            self.model.b = b.reshape(self.model.b.shape)
            Y, _ = self.model.fprop(self.X, **self.meta)
            c,_ = cost.fprop(Y, self.Y)
            return c

        def grad(b):
            self.model.b = b.reshape(self.model.b.shape)
            Y, _ = self.model.fprop(self.X, **self.meta)
            delta, _ = cost.bprop(Y, self.Y)
            [_, gb], _ = self.model.grads(self.X, delta, **self.meta)

            return gb.ravel()

        assert scipy.optimize.check_grad(func, grad, self.model.b.ravel()) < 1e-5