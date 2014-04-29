__author__ = 'mdenil'

import numpy as np

import pycuda.autoinit
from pycuda import cumath

from gpu import space
from gpu.model import layer

import generic.model.transfer

import gpu.conv

import scikits.cuda.linalg
scikits.cuda.linalg.init()

import gpu.utils

class Linear(generic.model.transfer.Linear, layer.Layer):
    def __init__(self, *args, **kwargs):
        super(Linear, self).__init__(*args, **kwargs)
        self.W = gpu.utils.cpu_to_gpu(self.W.astype(np.float32))

    def _fprop(self, X):
        Y = scikits.cuda.linalg.dot(X, self.W)
        Y_space = space.GPUSpace.infer(Y, ['b', 'd'])
        return Y, Y_space

    def _bprop(self, delta):
        return scikits.cuda.linalg.dot(delta, self.W, transb='T')

    def _grads(self, X, delta):
        return [scikits.cuda.linalg.dot(X, delta, transa='T')]


class Softmax(generic.model.transfer.Softmax, layer.Layer):
    def __init__(self, *args, **kwargs):
        super(Softmax, self).__init__(*args, **kwargs)

        self.W = gpu.utils.cpu_to_gpu(self.W.astype(np.float32))
        self.b = gpu.utils.cpu_to_gpu(self.b.astype(np.float32))
        self._b_space = space.GPUSpace.infer(self.b, ('b', 'w'))
        self._sum_vector_classes = gpu.utils.cpu_to_gpu(np.ones((self.n_classes,1), dtype=np.float32))

    def _fprop(self, X, X_space):
        A = scikits.cuda.linalg.dot(X, self.W)
        B, bias_space = self._b_space.broadcast(self.b, b=X_space.get_extent('b'))
        Y = cumath.exp(A + B)

        Z = scikits.cuda.linalg.dot(Y, self._sum_vector_classes)
        Z_space = bias_space.with_extents(w=1)
        Z, Z_space = Z_space.broadcast(Z, w=self.n_classes)

        Y /= Z

        return Y

    def _bprop(self, delta, Y):
        return scikits.cuda.linalg.dot(delta * Y * (1.0 - Y), self.W, transb='T')

    def _grads(self, delta, X, Y):
        delta *= Y * (1.0-Y)
        grad_W = scikits.cuda.linalg.dot(X, delta, transa='T')

        sum_vector_batch = pycuda.gpuarray.zeros((delta.shape[0],1), dtype=np.float32)
        sum_vector_batch += 1.0

        grad_b = scikits.cuda.linalg.dot(delta, sum_vector_batch, transa='T').reshape(self.b.shape)

        return [grad_W, grad_b]


class SentenceConvolution(generic.model.transfer.SentenceConvolution, layer.Layer):
    def __init__(self, *args, **kwargs):
        super(SentenceConvolution, self).__init__(*args, **kwargs)

        self.W = gpu.utils.cpu_to_gpu(self.W.astype(np.float32))
        self._kernel_space = space.GPUSpace.infer(self.W, ['f', 'd', 'c', 'w'])
        self.W, self._kernel_space = self._kernel_space.transform(self.W, [('b', 'f', 'd', 'c'), 'w'])

        self._conv = gpu.conv.FFTConv1D()

    def _fprop(self, X, X_space):
        K, _ = self._kernel_space.broadcast(gpu.utils.fliplr(self.W), b=X_space.get_extent('b'))
        X = self._conv.conv(X, K)

        X_space = X_space.with_extents(w=X.shape[1])

        X, X_space = gpu.utils.sum_along_axis(X, X_space, 'c')
        X, X_space = X_space.transform(X, (('b', 'd', 'f'), 'w'))

        return X, X_space

    def _bprop(self, delta, delta_space):
        K, _ = self._kernel_space.broadcast(self.W, b=delta_space.get_extent('b'))

        delta = self._conv.conv(delta, K, mode='valid')
        delta_space = delta_space.with_extents(w=delta.shape[1])

        delta, delta_space = gpu.utils.sum_along_axis(delta, delta_space, 'f')
        delta_space = delta_space.rename_axes(c='f')

        return delta, delta_space

    def _grads(self, delta, delta_space, X):
        grad_W = self._conv.conv(gpu.utils.fliplr(delta), X, mode='valid')
        grad_W_space = delta_space.with_extents(w=grad_W.shape[1])

        grad_W, grad_W_space = gpu.utils.sum_along_axis(grad_W, grad_W_space, 'b')
        grad_W, grad_W_space = grad_W_space.transform(grad_W, [('b', 'f', 'd', 'c'), 'w'])

        return [grad_W]


class Bias(generic.model.transfer.Bias, layer.Layer):
    def __init__(self, *args, **kwargs):
        super(Bias, self).__init__(*args, **kwargs)

        self.b = gpu.utils.cpu_to_gpu(self.b.astype(np.float32))
        self._b_space = space.GPUSpace.infer(self.b, ('f', 'd'))

    def _fprop(self, X, X_space):
        B, _ = self._b_space.transform(self.b, X_space.axes, w=X_space.get_extent('w'), b=X_space.get_extent('b'))
        return X + B

    # bprop is a no-op

    def _grads(self, delta, delta_space):
        delta, delta_space = gpu.utils.sum_along_axis(delta, delta_space, 'b')
        grad_b, grad_b_space = gpu.utils.sum_along_axis(delta, delta_space, 'w')
        return [grad_b]
