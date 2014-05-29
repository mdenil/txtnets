__author__ = 'mdenil'

import numpy as np
import scipy.optimize

import unittest
from cpu import space

import cpu.model.transfer
import cpu.model.cost

model = cpu.model


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
        self.Y = np.equal.outer(self.Y, np.arange(self.n_classes)).astype(self.X.dtype)

        self.X_space = space.CPUSpace.infer(self.X, ['w', 'f', 'd', 'b'])

        self.meta = {'lengths': np.zeros(b) + w, 'space_below': self.X_space}

        self.cost = model.cost.CrossEntropy()

    def test_fprop(self):
        actual, _, _ = self.layer.fprop(self.X, meta=dict(self.meta))

        X, X_space = self.X_space.transform(self.X, ('b', ('d', 'f', 'w')))

        expected = np.exp(np.dot(X, self.layer.W) + self.layer.b)
        expected /= np.sum(expected, axis=1, keepdims=True)

        assert np.allclose(actual, expected)

    def test_bprop(self):
        def func(x):
            x = x.reshape(self.X.shape)
            Y, meta, fprop_state = self.layer.fprop(x, meta=dict(self.meta))
            meta['space_below'] = meta['space_above']
            c, meta, cost_state = self.cost.fprop(Y, self.Y, meta=dict(meta))
            return c

        def grad(x):
            x = x.reshape(self.X.shape)
            Y, meta, fprop_state = self.layer.fprop(x, meta=dict(self.meta))
            meta['space_below'] = meta['space_above']
            cost, meta, cost_state = self.cost.fprop(Y, self.Y, meta=dict(meta))
            delta, meta = self.cost.bprop(Y, self.Y, meta=dict(meta), fprop_state=cost_state)
            meta['space_above'] = meta['space_below']
            delta, meta = self.layer.bprop(delta, meta=dict(meta), fprop_state=fprop_state)
            delta, _ = meta['space_below'].transform(delta, self.X_space.axes)
            return delta.ravel()

        assert scipy.optimize.check_grad(func, grad, self.X.ravel()) < 1e-5

    def test_grad_W(self):
        def func(w):
            self.layer.W = w.reshape(self.layer.W.shape)
            Y, meta, fprop_state = self.layer.fprop(self.X, meta=dict(self.meta))
            meta['space_below'] = meta['space_above']
            c, meta, cost_state = self.cost.fprop(Y, self.Y, meta=dict(meta))
            return c

        def grad(w):
            self.layer.W = w.reshape(self.layer.W.shape)
            Y, meta, fprop_state = self.layer.fprop(self.X, meta=dict(self.meta))
            meta['space_below'] = meta['space_above']
            cost, meta, cost_state = self.cost.fprop(Y, self.Y, meta=meta)
            delta, meta = self.cost.bprop(Y, self.Y, meta=dict(meta), fprop_state=cost_state)
            meta['space_above'] = meta['space_below']
            [grad_W, _] = self.layer.grads(delta, meta=dict(meta), fprop_state=fprop_state)

            return grad_W.ravel()

        assert scipy.optimize.check_grad(func, grad, self.layer.W.ravel()) < 1e-5

    def test_grad_b(self):
        cost = model.cost.CrossEntropy()

        def func(b):
            self.layer.b = b.reshape(self.layer.b.shape)
            Y, meta, fprop_state = self.layer.fprop(self.X, meta=dict(self.meta))
            meta['space_below'] = meta['space_above']
            c, meta, cost_state = cost.fprop(Y, self.Y, meta=dict(meta))
            return c

        def grad(b):
            self.layer.b = b.reshape(self.layer.b.shape)
            Y, meta, fprop_state = self.layer.fprop(self.X, meta=dict(self.meta))
            meta['space_below'] = meta['space_above']
            c, meta, cost_state = cost.fprop(Y, self.Y, meta=dict(meta))
            delta, meta = cost.bprop(Y, self.Y, meta=dict(meta), fprop_state=cost_state)
            meta['space_above'] = meta['space_below']
            [_, grad_b] = self.layer.grads(delta, meta=dict(meta), fprop_state=fprop_state)

            return grad_b.ravel()

        assert scipy.optimize.check_grad(func, grad, self.layer.b.ravel()) < 1e-5



class Bias(unittest.TestCase):
    def setUp(self):
        b,w,f,d = 2, 1, 3, 2

        self.layer = model.transfer.Bias(
            n_feature_maps=f,
            n_input_dims=d)
        # biases default to zero, lets mix it up a bit
        self.layer.b = np.random.standard_normal(size=self.layer.b.shape)

        self.X = np.random.standard_normal(size=(b, w, f, d))
        self.X_space = space.CPUSpace.infer(self.X, ('b', 'w', 'f', 'd'))
        self.meta = {'lengths': np.zeros(b) + w, 'space_below': self.X_space}


    def test_fprop(self):
        actual, meta, fprop_state = self.layer.fprop(self.X, meta=dict(self.meta))
        X, _ = self.X_space.transform(self.X, ('b', 'd', 'f', 'w'))
        expected = X + self.layer.b[np.newaxis, :, :, np.newaxis]

        assert np.allclose(actual, expected)

    def test_bprop(self):
        def func(x):
            X = x.reshape(self.X.shape)
            Y, meta, fprop_state = self.layer.fprop(X, meta=dict(self.meta))
            return Y.sum()

        def grad(x):
            X = x.reshape(self.X.shape)
            print self.meta['space_below'], X.shape
            Y, meta, fprop_state = self.layer.fprop(X, meta=dict(self.meta))
            delta, meta = self.layer.bprop(np.ones_like(Y), meta=dict(meta), fprop_state=fprop_state)
            print meta['space_below'], delta.shape, self.X_space.axes
            delta, _ = meta['space_below'].transform(delta, self.X_space.axes)

            return delta.ravel()

        assert scipy.optimize.check_grad(func, grad, self.X.ravel()) < 1e-5

    def test_grad_b(self):
        def func(b):
            self.layer.b = b.reshape(self.layer.b.shape)
            Y, meta, fprop_state = self.layer.fprop(self.X, meta=dict(self.meta))
            return Y.sum()

        def grad(b):
            self.layer.b = b.reshape(self.layer.b.shape)

            Y, meta, fprop_state = self.layer.fprop(self.X, meta=dict(self.meta))
            grads = self.layer.grads(np.ones_like(Y), meta=dict(meta), fprop_state=fprop_state)

            gb = grads[0]

            return gb.ravel()

        assert scipy.optimize.check_grad(func, grad, self.layer.b.ravel()) < 1e-5



class SentenceConvolution(unittest.TestCase):
    def setUp(self):
        b,w,f,d,c = 2, 20, 2, 2, 2
        kernel_width = 4

        self.layer = model.transfer.SentenceConvolution(
            n_feature_maps=f,
            n_input_dimensions=d,
            n_channels=c,
            kernel_width=kernel_width)

        self.X = np.random.standard_normal(size=(b,w,d,c))

        # features in the X_space are channels in the convolution layer
        self.X_space = space.CPUSpace.infer(self.X, ['b', 'w', 'd', 'f'])
        self.meta = {'lengths': np.random.randint(1, w, size=b), 'space_below': self.X_space}

        # Using this causes test_grad_W to fail if you forget to flip delta before the convolution when computing
        # the gradient (this is good because if you forget that you're doing it wrong).  If you don't have a mask and
        # just backprop all ones then the test still passes without the flip (i.e. with the wrong gradient).
        self.delta_mask = np.random.uniform(size=(b*d*f, w+kernel_width-1)) > 0.5


    def test_fprop(self):
        self.skipTest('WRITEME')

    def test_bprop(self):
        def func(x):
            X = x.reshape(self.X.shape)
            Y, meta, fprop_state = self.layer.fprop(X, meta=dict(self.meta))
            Y *= self.delta_mask
            return Y.sum()

        def grad(x):
            X = x.reshape(self.X.shape)
            Y, meta, fprop_state = self.layer.fprop(X, meta=dict(self.meta))
            delta, meta = self.layer.bprop(self.delta_mask, meta=dict(meta), fprop_state=fprop_state)
            delta, _ = meta['space_below'].transform(delta, self.X_space.axes)
            return delta.ravel()

        assert scipy.optimize.check_grad(func, grad, self.X.ravel()) < 1e-5

    def test_grad_W(self):
        def func(W):
            self.layer.W = W.reshape(self.layer.W.shape)
            Y, meta, fprop_state = self.layer.fprop(self.X.copy(), meta=dict(self.meta))
            Y *= self.delta_mask
            return Y.sum()

        def grad(W):
            self.layer.W = W.reshape(self.layer.W.shape)

            Y, meta, fprop_state = self.layer.fprop(self.X.copy(), meta=dict(self.meta))
            [grad_W] = self.layer.grads(self.delta_mask, meta=dict(meta), fprop_state=fprop_state)

            return grad_W.ravel()

        assert scipy.optimize.check_grad(func, grad, self.layer.W.ravel()) < 1e-5


class Linear(unittest.TestCase):
    def setUp(self):
        b,w,f,d = 2, 20, 2, 2
        kernel_width = 4

        self.layer = model.transfer.Linear(
            n_input=f*d*w,
            n_output=20)

        self.X = np.random.standard_normal(size=(b,w,d,f))

        self.X_space = space.CPUSpace.infer(self.X, ['b', 'w', 'd', 'f'])
        self.meta = {'lengths': np.random.randint(1, w, size=b), 'space_below': self.X_space}

        self.delta_mask = np.random.uniform(size=(b, 20)) > 0.5


    def test_fprop(self):
        self.skipTest('WRITEME')

    def test_bprop(self):
        def func(x):
            X = x.reshape(self.X.shape)
            Y, meta, fprop_state = self.layer.fprop(X, meta=dict(self.meta))
            Y *= self.delta_mask
            return Y.sum()

        def grad(x):
            X = x.reshape(self.X.shape)
            Y, meta, fprop_state = self.layer.fprop(X, meta=dict(self.meta))
            delta, meta = self.layer.bprop(self.delta_mask, meta=dict(meta), fprop_state=fprop_state)
            delta, _ = meta['space_below'].transform(delta, self.X_space.axes)
            return delta.ravel()

        assert scipy.optimize.check_grad(func, grad, self.X.ravel()) < 1e-5

    def test_grad_W(self):
        def func(W):
            self.layer.W = W.reshape(self.layer.W.shape)
            Y, meta, fprop_state = self.layer.fprop(self.X.copy(), meta=dict(self.meta))
            Y *= self.delta_mask
            return Y.sum()

        def grad(W):
            self.layer.W = W.reshape(self.layer.W.shape)

            Y, meta, fprop_state = self.layer.fprop(self.X.copy(), meta=dict(self.meta))
            delta = np.ones_like(Y)
            [grad_W] = self.layer.grads(self.delta_mask, meta=dict(meta), fprop_state=fprop_state)

            return grad_W.ravel()

        assert scipy.optimize.check_grad(func, grad, self.layer.W.ravel()) < 1e-5


class ReshapeForDocuments(unittest.TestCase):
    def setUp(self):
        b, d, f, w = 15, 1, 3, 14
        self.padded_sentence_length = 5
        self.model = model.transfer.ReshapeForDocuments()

        self.X = np.random.standard_normal(size=(b, d, f, w))
        self.X_space = space.CPUSpace.infer(self.X, ('b', 'd', 'f', 'w'))

        self.meta = {
            'lengths': np.random.randint(1, w, size=b),
            'space_below': self.X_space,
            'padded_sentence_length': self.padded_sentence_length,
            'lengths2': np.random.randint(1, self.padded_sentence_length, size=b // self.padded_sentence_length)
        }

    def test_bprop(self):
        Y, meta, state = self.model.fprop(self.X, meta=dict(self.meta))
        X, meta = self.model.bprop(Y, meta=dict(meta), fprop_state=state)

        X, _ = meta['space_below'].transform(X, self.X_space.axes)

        self.assertTrue(np.allclose(X, self.X))