__author__ = 'mdenil'

import numpy as np
import scipy.optimize

import unittest
from cpu import space

import cpu.model.transfer
import cpu.model.nonlinearity
import cpu.model.cost
import cpu.model.model

model = cpu.model

class Tanh(unittest.TestCase):
    def setUp(self):
        w,f,d,b = 2, 3, 5, 10
        self.n_classes = 7
        self.X = np.random.standard_normal(size=(w, f, d, b))
        self.X_space = space.CPUSpace.infer(self.X, ['w', 'f', 'd', 'b'])
        self.Y = np.random.randint(0, self.n_classes, size=b)
        self.Y = np.equal.outer(self.Y, np.arange(self.n_classes)).astype(self.X.dtype)

        self.layer = model.nonlinearity.Tanh()
        self.meta = {'space_below': self.X_space, 'lengths': np.zeros(b) + w}

        self.csm = model.model.CSM(
            layers=[
                self.layer,
                model.transfer.Softmax(
                    n_classes=self.n_classes,
                    n_input_dimensions=w*f*d),
                ])
        self.cost = model.cost.CrossEntropy()

    def test_fprop(self):
        actual, _, _ = self.layer.fprop(self.X, meta=self.meta)
        expected = np.tanh(self.X)

        assert np.allclose(actual, expected)

    def test_bprop(self):
        def func(x):
            x = x.reshape(self.X.shape)
            Y, meta, fprop_state = self.csm.fprop(x, meta=self.meta, return_state=True)
            c, meta, cost_state = self.cost.fprop(Y, self.Y, meta=meta)
            return c

        def grad(x):
            X = x.reshape(self.X.shape)
            Y, meta, fprop_state = self.csm.fprop(X, meta=self.meta, return_state=True)
            meta['space_below'] = meta['space_above']
            c, meta, cost_state = self.cost.fprop(Y, self.Y, meta=meta)

            delta, meta = self.cost.bprop(Y, self.Y, meta=meta, fprop_state=cost_state)
            delta = self.csm.bprop(delta, meta=dict(meta), fprop_state=fprop_state)
            return delta.ravel()

        assert scipy.optimize.check_grad(func, grad, self.X.ravel()) < 1e-5


class Relu(unittest.TestCase):
    def setUp(self):
        w,f,d,b = 2, 3, 5, 10
        self.n_classes = 7
        self.X = np.random.standard_normal(size=(w, f, d, b))
        self.X_space = space.CPUSpace.infer(self.X, ['w', 'f', 'd', 'b'])
        self.Y = np.random.randint(0, self.n_classes, size=b)
        self.Y = np.equal.outer(self.Y, np.arange(self.n_classes)).astype(self.X.dtype)

        self.layer = model.nonlinearity.Relu()
        self.meta = {'space_below': self.X_space, 'lengths': np.zeros(b) + w}

        self.csm = model.model.CSM(
            layers=[
                self.layer,
                model.transfer.Softmax(
                    n_classes=self.n_classes,
                    n_input_dimensions=w*f*d),
                ])
        self.cost = model.cost.CrossEntropy()

    def test_fprop(self):
        actual, _, _ = self.layer.fprop(self.X, meta=self.meta)
        expected = np.maximum(0, self.X)

        assert np.allclose(actual, expected)

    def test_bprop(self):
        def func(x):
            x = x.reshape(self.X.shape)
            Y, meta, fprop_state = self.csm.fprop(x, meta=self.meta, return_state=True)
            c, meta, cost_state = self.cost.fprop(Y, self.Y, meta=meta)
            return c

        def grad(x):
            X = x.reshape(self.X.shape)
            Y, meta, fprop_state = self.csm.fprop(X, meta=self.meta, return_state=True)
            meta['space_below'] = meta['space_above']
            c, meta, cost_state = self.cost.fprop(Y, self.Y, meta=meta)
            delta, meta = self.cost.bprop(Y, self.Y, meta=meta, fprop_state=cost_state)
            delta = self.csm.bprop(delta, meta=dict(meta), fprop_state=fprop_state)
            return delta.ravel()

        assert scipy.optimize.check_grad(func, grad, self.X.ravel()) < 1e-5

