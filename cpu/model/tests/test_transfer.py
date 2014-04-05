__author__ = 'mdenil'

import numpy as np
import scipy.optimize

import unittest
from cpu import model
from cpu import space

class Softmax(unittest.TestCase):
    def setUp(self):
        # X = ['w', 'f', 'd', 'b']
        # Y = ['d', 'b'] (d = classes)
        w,f,d,b = 2, 3, 5, 10
        self.n_input_dimensions = w*f*d
        self.n_classes = 7

        self.layer = model.transfer.Softmax(
            n_classes=self.n_classes,
            n_input_dimensions=self.n_input_dimensions)

        self.X = np.random.standard_normal(size=(w, f, d, b))
        self.Y = np.random.randint(0, self.n_classes, size=b)
        self.Y = np.equal.outer(np.arange(self.n_classes), self.Y).astype(self.X.dtype)

        self.X_space = space.Space.infer(self.X, ['w', 'f', 'd', 'b'])

        self.meta = {'lengths': np.zeros(b) + w, 'space_below': self.X_space}

    def test_fprop(self):
        actual, _, _ = self.layer.fprop(self.X, meta=self.meta)
        expected = np.exp(np.dot(self.layer.W, self.X.reshape((self.n_input_dimensions, -1))) + self.layer.b)
        expected /= np.sum(expected, axis=0)

        assert np.allclose(actual, expected)

    def test_bprop(self):
        cost = model.cost.CrossEntropy()

        def func(x):
            x = x.reshape(self.X.shape)
            Y, _, _ = self.layer.fprop(x, meta=self.meta)
            c,_ = cost.fprop(Y, self.Y)
            return c

        def grad(x):
            X = x.reshape(self.X.shape)
            Y, meta, fprop_state = self.layer.fprop(X, self.meta)
            delta, _ = cost.bprop(Y, self.Y)
            delta, _ = self.layer.bprop(X, delta, meta=meta, fprop_state=fprop_state)
            return delta.ravel()

        assert scipy.optimize.check_grad(func, grad, self.X.ravel()) < 1e-5

    def test_grad_W(self):
        cost = model.cost.CrossEntropy()

        def func(w):
            self.layer.W = w.reshape(self.layer.W.shape)
            Y, _, _ = self.layer.fprop(self.X, meta=self.meta)
            c,_ = cost.fprop(Y, self.Y)
            return c

        def grad(w):
            self.layer.W = w.reshape(self.layer.W.shape)
            Y, meta, fprop_state = self.layer.fprop(self.X, meta=self.meta)
            delta, _ = cost.bprop(Y, self.Y)
            [grad_W, _] = self.layer.grads(self.X, delta, meta=meta, fprop_state=fprop_state)

            return grad_W.ravel()

        assert scipy.optimize.check_grad(func, grad, self.layer.W.ravel()) < 1e-5

    def test_grad_b(self):
        cost = model.cost.CrossEntropy()

        def func(b):
            self.layer.b = b.reshape(self.layer.b.shape)
            Y, _, _ = self.layer.fprop(self.X, meta=self.meta)
            c,_ = cost.fprop(Y, self.Y)
            return c

        def grad(b):
            self.layer.b = b.reshape(self.layer.b.shape)
            Y, meta, fprop_state = self.layer.fprop(self.X, meta=self.meta)
            delta, _ = cost.bprop(Y, self.Y)
            [_, grad_b] = self.layer.grads(self.X, delta, meta=meta, fprop_state=fprop_state)

            return grad_b.ravel()

        assert scipy.optimize.check_grad(func, grad, self.layer.b.ravel()) < 1e-5



class Bias(unittest.TestCase):
    def setUp(self):
        b,w,f,d = 1, 1, 3, 5
        self.n_classes = 7

        self.layer = model.transfer.Bias(
            n_feature_maps=f,
            n_input_dims=d)
        # biases default to zero
        self.layer.b = np.random.standard_normal(size=self.layer.b.shape)

        self.X = np.random.standard_normal(size=(b,w,f,d))
        self.Y = np.random.randint(0, self.n_classes, size=b)
        self.Y = np.equal.outer(np.arange(self.n_classes), self.Y).astype(self.X.dtype)

        self.X_space = space.Space.infer(self.X, ['b', 'w', 'f', 'd'])
        self.meta = {'lengths': np.zeros(b) + w, 'space_below': self.X_space}

        self.csm = model.model.CSM(
            input_axes=['b', 'w', 'f', 'd'],
            layers=[
                self.layer,
                model.transfer.Softmax(
                    n_classes=self.n_classes,
                    n_input_dimensions=w*f*d),
                ])
        self.cost = model.cost.CrossEntropy()


    def test_fprop(self):
        actual, _ = self.layer.fprop(self.X.copy(), meta=self.meta)
        expected = self.X + self.layer.b

        assert np.allclose(actual, expected)

    def test_bprop(self):
        def func(x):
            x = x.reshape(self.X.shape)
            Y = self.csm.fprop(x.copy(), meta=self.meta)
            c,_ = self.cost.fprop(Y.copy(), self.Y.copy())
            return c

        def grad(x):
            X = x.reshape(self.X.shape)
            Y, meta, fprop_state = self.csm.fprop(X.copy(), meta=self.meta, return_meta=True, return_state=True)
            delta, _ = self.cost.bprop(Y.copy(), self.Y.copy())
            delta = self.csm.bprop(delta, fprop_state=fprop_state)
            return delta.ravel()

        assert scipy.optimize.check_grad(func, grad, self.X.ravel()) < 1e-5

    def test_grad_b(self):

        def func(b):
            self.layer.b = b.reshape(self.layer.b.shape)
            Y, _ = self.layer.fprop(self.X.copy(), meta=self.meta)
            c = Y.sum()
            return Y.sum()

        def grad(b):
            self.layer.b = b.reshape(self.layer.b.shape)

            Y, meta = self.layer.fprop(self.X.copy(), meta=self.meta)
            delta = np.ones_like(Y)
            [gb] = self.layer.grads(self.X.copy(), delta, meta=meta)

            return gb.ravel()

        assert scipy.optimize.check_grad(func, grad, self.layer.b.ravel()) < 1e-5

