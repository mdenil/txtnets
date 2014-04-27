__author__ = 'mdenil'

import numpy as np

import pycuda.autoinit
from pycuda import cumath

from gpu import space
from gpu.model import layer

import scikits.cuda.linalg
scikits.cuda.linalg.init()

import gpu.utils

class Softmax(layer.Layer):
    def __init__(self,
                 n_classes,
                 n_input_dimensions):
        self.n_classes = n_classes
        self.n_input_dimensions = n_input_dimensions

        self.W = 0.1 * np.random.standard_normal(size=(self.n_input_dimensions, self.n_classes))
        self.b = np.zeros(shape=(1, self.n_classes))

        self.W = gpu.utils.cpu_to_gpu(self.W.astype(np.float32))
        self.b = gpu.utils.cpu_to_gpu(self.b.astype(np.float32))
        self._b_space = space.GPUSpace.infer(self.b, ('b', 'w'))
        self._sum_vector_classes = gpu.utils.cpu_to_gpu(np.ones((self.n_classes,1), dtype=np.float32))

    def fprop(self, X, meta):

        X, X_space = meta['space_below'].transform(X, ('b', ('w', 'f', 'd')))

        if not X.shape[1] == self.W.shape[0]:
            raise ValueError("Cannot multiply X.shape={} ({}) with W.shape={}".format(X.shape, X_space, self.W.shape))

        A = scikits.cuda.linalg.dot(X, self.W)
        B, bias_space = self._b_space.broadcast(self.b, b=X_space.get_extent('b'))
        Y = cumath.exp(A + B)

        Z = scikits.cuda.linalg.dot(Y, self._sum_vector_classes)
        Z_space = bias_space.with_extents(w=1)
        Z, Z_space = Z_space.broadcast(Z, w=self.n_classes)

        Y /= Z

        Y_space = X_space.without_axes(('w', 'f'))
        Y_space = Y_space.with_extents(d=self.n_classes)
        assert Y_space.is_compatible_shape(Y)

        lengths_below = meta['lengths']
        meta['lengths'] = np.ones_like(meta['lengths'])
        meta['space_above'] = Y_space

        fprop_state = {
            'X_space': X_space,
            'Y_space': Y_space,
            'lengths_below': lengths_below.copy(),
            'X': X,
            'Y': Y,
        }

        return Y, meta, fprop_state

    def bprop(self, delta, meta, fprop_state):
        Y = fprop_state['Y']

        assert fprop_state['Y_space'].is_compatible_shape(Y)

        delta, delta_space = meta['space_above'].transform(delta, ['b', 'd'])

        # out = np.dot(delta * Y * (1-Y), self.W.T)
        out = scikits.cuda.linalg.dot(delta * Y * (1.0 - Y), self.W, transb='T')

        meta['space_below'] = fprop_state['X_space']
        meta['lengths'] = fprop_state['lengths_below']
        return out, meta

    def grads(self, delta, meta, fprop_state):
        X = fprop_state['X']
        Y = fprop_state['Y']
        X_space = fprop_state['X_space']
        X, X_space = X_space.transform(X, ('b', ('w', 'f', 'd')))

        delta, delta_space = meta['space_above'].transform(delta, ('b', ('w', 'f', 'd')))

        delta *= Y * (1.0-Y)

        # grad_W = np.dot(X.T, delta)
        grad_W = scikits.cuda.linalg.dot(X, delta, transa='T')


        # grad_b = delta.sum(axis=0).reshape(self.b.shape)
        # sum_vector_batch = gpu.utils.cpu_to_gpu(np.ones((delta_space.get_extent('b'),1), dtype=np.float32))

        sum_vector_batch = pycuda.gpuarray.zeros((delta_space.get_extent('b'),1), dtype=np.float32)
        sum_vector_batch += 1.0

        grad_b = scikits.cuda.linalg.dot(delta, sum_vector_batch, transa='T').reshape(self.b.shape)


        return [grad_W, grad_b]

    def params(self):
        return [self.W, self.b]

    def __repr__(self):
        return "{}(W={})".format(
            self.__class__.__name__,
            self.W.shape)