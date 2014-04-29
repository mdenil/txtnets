__author__ = 'mdenil'

import numpy as np

from cpu import space
from cpu import conv
from cpu.model import layer

import generic.model.transfer

class Linear(generic.model.transfer.Linear, layer.Layer):

    def _fprop(self, X):
        Y = np.dot(X, self.W)
        Y_space = space.CPUSpace.infer(Y, ['b', 'd'])
        return Y, Y_space

    def _bprop(self, delta):
        return np.dot(delta, self.W.T)

    def _grads(self, X, delta):
        return [np.dot(X.T, delta)]


class Softmax(generic.model.transfer.Softmax, layer.Layer):

    def _fprop(self, X, X_space):
        Y = np.exp(np.dot(X, self.W) + self.b)
        Y /= np.sum(Y, axis=1, keepdims=True)
        return Y

    def _bprop(self, delta, Y):
        return np.dot(delta * Y * (1-Y), self.W.T)

    def _grads(self, delta, X, Y):
        delta = delta * Y * (1-Y)
        grad_W = np.dot(X.T, delta)
        grad_b = delta.sum(axis=0).reshape(self.b.shape)
        return [grad_W, grad_b]


class SentenceConvolution(generic.model.transfer.SentenceConvolution, layer.Layer):
    def __init__(self, *args, **kwargs):
        super(SentenceConvolution, self).__init__(*args, **kwargs)

        self._kernel_space = space.CPUSpace.infer(self.W, ['f', 'd', 'c', 'w'])
        self.W, self._kernel_space = self._kernel_space.transform(self.W, [('b', 'f', 'd', 'c'), 'w'])

    def _fprop(self, X, X_space):
        K, _ = self._kernel_space.broadcast(np.fliplr(self.W), b=X_space.get_extent('b'))
        X = conv.fftconv1d(X, K, n_threads=self.n_threads)

        X_space = X_space.with_extents(w=X.shape[1])

        X, X_space = X_space.transform(X, (('b', 'd', 'f'), 'w', 'c'))
        X = X.sum(axis=X_space.axes.index('c'))
        X_space = X_space.without_axes('c')

        return X, X_space

    def _bprop(self, delta, delta_space):
        K, _ = self._kernel_space.broadcast(self.W, b=delta_space.get_extent('b'))

        delta = conv.fftconv1d(delta, K, n_threads=self.n_threads, mode='valid')
        delta_space = delta_space.with_extents(w=delta.shape[1])

        delta, delta_space = delta_space.transform(delta, delta_space.folded_axes)
        delta = delta.sum(axis=delta_space.axes.index('f'))
        delta_space = delta_space.without_axes('f')
        delta_space = delta_space.rename_axes(c='f')

        return delta, delta_space

    def _grads(self, delta, delta_space, X):
        grad_W = conv.fftconv1d(np.fliplr(delta), X, n_threads=self.n_threads, mode='valid')
        grad_W_space = delta_space.with_extents(w=grad_W.shape[1])

        grad_W, grad_W_space = grad_W_space.transform(grad_W, grad_W_space.folded_axes)

        grad_W = grad_W.sum(axis=grad_W_space.axes.index('b'))
        grad_W_space = grad_W_space.without_axes('b')
        grad_W, grad_W_space = grad_W_space.transform(grad_W, [('b', 'f', 'd', 'c'), 'w'])

        return [grad_W]


class Bias(generic.model.transfer.Bias, layer.Layer):

    def _fprop(self, X, X_space):
        return X + self.b

    # bprop is a no-op

    def _grads(self, delta, delta_space):
        delta, working_space = delta_space.transform(delta, ['f', 'd', ('b', 'w')])
        grad_b = delta.sum(axis=2)
        return [grad_b]
