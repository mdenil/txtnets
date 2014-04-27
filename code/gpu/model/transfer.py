__author__ = 'mdenil'

import numpy as np

import pycuda.autoinit
from pycuda import cumath

from gpu import space
from gpu.model import layer

import gpu.conv

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


class SentenceConvolution(layer.Layer):
    def __init__(self,
                 n_feature_maps,
                 kernel_width,
                 n_input_dimensions,
                 n_channels,
                 ):

        self.n_feature_maps = n_feature_maps
        self.kernel_width = kernel_width
        self.n_input_dimensions = n_input_dimensions
        self.n_channels = n_channels

        self.W = 0.1 * np.random.standard_normal(
            size=(self.n_feature_maps, self.n_input_dimensions, self.n_channels, self.kernel_width))
        self.W = self.W.astype(np.float32)
        self.W = gpu.utils.cpu_to_gpu(self.W)

        self._kernel_space = space.GPUSpace.infer(self.W, ['f', 'd', 'c', 'w'])
        self.W, self._kernel_space = self._kernel_space.transform(self.W, [('b', 'f', 'd', 'c'), 'w'])

        self._conv = gpu.conv.FFTConv1D()

    def fprop(self, X, meta):

        # Things go wrong if the w extent of X is smaller than the kernel width... for now just don't do that.
        if meta['space_below'].get_extent('w') < self.kernel_width:
            raise ValueError("SentenceConvolution does not support input with w={} extent smaller than kernel_width={}".format(
                meta['space_below'].get_extent('w'),
                self.kernel_width
            ))

        working_space = meta['space_below']
        lengths = meta['lengths']

        # features in the input space become channels here
        if 'f' in working_space.folded_axes:
            working_space = working_space.rename_axes(f='c')
        else:
            X, working_space = working_space.add_axes(X, 'c')

        fprop_state = {
            'input_space': working_space,
            'X': X,
            'lengths_below': lengths.copy()
        }

        b, d, c, w = working_space.get_extents(['b','d','c','w'])

        if not self.n_channels == c:
            raise ValueError("n_chanels={} but the data has {} channels.".format(self.n_channels, c))
        if not self.n_input_dimensions == d:
            raise ValueError("n_input_dimensions={} but the data has {} dimensions.".format(self.n_input_dimensions, d))
        f = self.n_feature_maps

        X, working_space = working_space.transform(X, [('b', 'f', 'd', 'c'), 'w'])
        X, working_space = working_space.broadcast(X, f=f)

        K, _ = self._kernel_space.broadcast(gpu.utils.fliplr(self.W), b=b)

        X = self._conv.conv(X, K)

        representation_length = X.shape[1]

        # length of a wide convolution
        lengths = lengths + self.kernel_width - 1

        working_space = working_space.with_extents(w=representation_length)

        X, working_space = gpu.utils.sum_along_axis(X, working_space, 'c')
        X, working_space = working_space.transform(X, (('b', 'd', 'f'), 'w'))

        meta['space_above'] = working_space
        meta['lengths'] = lengths

        return X, meta, fprop_state

    def bprop(self, delta, meta, fprop_state):
        working_space = meta['space_above']
        lengths = meta['lengths']
        X_space = fprop_state['input_space']

        delta, working_space = working_space.transform(delta, [('b','f','d','c'), 'w'], c=X_space.get_extent('c'))
        K, _ = self._kernel_space.broadcast(self.W, b=working_space.get_extent('b'))

        delta =  self._conv.conv(delta, K, mode='valid')
        working_space = working_space.with_extents(w=delta.shape[1])

        lengths = lengths - self.kernel_width + 1

        delta, working_space = gpu.utils.sum_along_axis(delta, working_space, 'f')
        working_space = working_space.rename_axes(c='f')

        meta['space_below'] = working_space
        meta['lengths'] = lengths

        assert np.all(lengths == fprop_state['lengths_below'])

        assert list(working_space.shape) == list(delta.shape)

        return delta, meta

    def grads(self, delta, meta, fprop_state):
        delta_space = meta['space_above']
        X = fprop_state['X']
        X_space = fprop_state['input_space']

        delta, delta_space = delta_space.transform(delta, [('b','f','d','c'), 'w'], c=X_space.get_extent('c'))
        X, X_space = X_space.transform(X, [('b','f','d','c'), 'w'], f=delta_space.get_extent('f'))

        grad_W = self._conv.conv(gpu.utils.fliplr(delta), X, mode='valid')
        grad_W_space = delta_space.with_extents(w=grad_W.shape[1])

        grad_W, grad_W_space = grad_W_space.transform(grad_W, grad_W_space.folded_axes)

        grad_W, grad_W_space = gpu.utils.sum_along_axis(grad_W, grad_W_space, 'b')
        # grad_W = grad_W.sum(axis=grad_W_space.axes.index('b'))
        # grad_W_space = grad_W_space.without_axes('b')

        grad_W, grad_W_space = grad_W_space.transform(grad_W, [('b','f','d','c'), 'w'])

        return [grad_W]

    def params(self):
        return [self.W]

    def __repr__(self):
        return "{}(W={})".format(
            self.__class__.__name__,
            self.W.shape)
