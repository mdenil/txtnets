__author__ = 'mdenil'

import numpy as np
import psutil

from cpu import space
from cpu import conv
from cpu.model import layer

class Softmax(layer.Layer):
    def __init__(self,
                 n_classes,
                 n_input_dimensions):
        self.n_classes = n_classes
        self.n_input_dimensions = n_input_dimensions

        self.W = 0.05 * np.random.standard_normal(size=(self.n_classes, self.n_input_dimensions))
        self.b = np.zeros(shape=(self.n_classes,1))

    def fprop(self, X, meta):
        X, X_space = meta['space_below'].transform(X, ['wfd', 'b'])

        Y = np.exp(np.dot(self.W, X) + self.b)
        Y /= np.sum(Y, axis=0)

        Y_space = X_space.without_axes('wf')
        Y_space = Y_space.with_extent(d=self.n_classes)

        meta['lengths'] = np.zeros(meta['lengths'].shape) + self.n_classes
        meta['space_above'] = Y_space

        fprop_state = {
            'X_space': X_space,
            'X': X,
            'Y': Y,
        }

        return Y, meta, fprop_state

    def bprop(self, delta, meta, fprop_state):
        Y = fprop_state['Y']

        out = np.dot(self.W.T, delta * Y * (1-Y))

        meta['space_below'] = fprop_state['X_space']
        return out, meta

    def grads(self, delta, meta, fprop_state):
        X = fprop_state['X']
        Y = fprop_state['Y']
        X_space = fprop_state['X_space']
        X, X_space = X_space.transform(X, ['wfd', 'b'])

        delta, delta_space = meta['space_above'].transform(delta, ['wfd', 'b'])

        delta = delta * Y * (1-Y)

        grad_W = np.dot(delta, X.T)
        grad_b = delta.sum(axis=1).reshape(self.b.shape)

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
                 n_threads=psutil.NUM_CPUS,
                 ):

        self.n_feature_maps = n_feature_maps
        self.kernel_width = kernel_width
        self.n_input_dimensions = n_input_dimensions
        self.n_threads = n_threads

        self.W = 0.05 * np.random.standard_normal(
            size=(self.n_feature_maps, self.n_input_dimensions, self.kernel_width))
        self._kernel_space = space.Space.infer(self.W, ['f', 'd', 'w'])
        self.W, self._kernel_space = self._kernel_space.transform(self.W, ['bfd', 'w'])

    def fprop(self, X, meta):
        working_space = meta['space_below']
        lengths = meta['lengths']

        fprop_state = {
            'input_space': working_space,
            'X': X,
        }

        b, d, w = working_space.get_extents(['b','d','w'])

        assert self.n_input_dimensions == d
        f = self.n_feature_maps

        X, working_space = working_space.transform(X, ['bfd', 'w'])
        X, working_space = working_space.broadcast(X, f=f)

        K, _ = self._kernel_space.broadcast(np.fliplr(self.W), b=b)

        X = conv.fftconv1d(X, K, n_threads=self.n_threads)

        representation_length = X.shape[1]

        # length of a wide convolution
        lengths = lengths + self.kernel_width - 1

        working_space = working_space.with_extent(w=representation_length)

        meta['space_above'] = working_space
        meta['lengths'] = lengths

        return X, meta, fprop_state

    def bprop(self, delta, meta, fprop_state):
        working_space = meta['space_above']
        lengths = meta['lengths']

        delta, working_space = working_space.transform(delta, ['bfd', 'w'])
        K, _ = self._kernel_space.broadcast(self.W, b=working_space.get_extent('b'))

        delta = conv.fftconv1d(delta, K, n_threads=self.n_threads, mode='valid')
        working_space = working_space.with_extent(w=delta.shape[1])

        lengths = lengths - self.kernel_width + 1

        delta, working_space = working_space.transform(delta, working_space.folded_axes)
        delta = delta.sum(axis=working_space.axes.index('f'))
        working_space = working_space.without_axes('f')

        meta['space_below'] = working_space
        meta['lengths'] = lengths

        assert list(working_space.shape) == list(delta.shape)

        return delta, meta

    def grads(self, delta, meta, fprop_state):
        delta_space = meta['space_above']
        X = fprop_state['X']
        X_space = fprop_state['input_space']

        delta, delta_space = delta_space.transform(delta, ['bfd', 'w'])
        X, X_space = X_space.transform(X, ['bfd', 'w'], f=delta_space.get_extent('f'))

        grad_W = conv.fftconv1d(np.fliplr(delta), X, n_threads=self.n_threads, mode='valid')
        grad_W_space = delta_space.with_extent(w=grad_W.shape[1])

        grad_W, grad_W_space = grad_W_space.transform(grad_W, grad_W_space.folded_axes)

        grad_W = grad_W.sum(axis=grad_W_space.axes.index('b'))
        grad_W_space = grad_W_space.without_axes('b')

        grad_W, grad_W_space = grad_W_space.transform(grad_W, ['fd', 'w'])

        return [grad_W]

    def params(self):
        return [self.W]

    def __repr__(self):
        return "{}(W={})".format(
            self.__class__.__name__,
            self.W.shape)


class Bias(layer.Layer):
    def __init__(self, n_input_dims, n_feature_maps):
        self.n_input_dims = n_input_dims
        self.n_feature_maps = n_feature_maps

        self.b = np.zeros((n_feature_maps, n_input_dims))

    def fprop(self, X, meta):
        working_space = meta['space_below']

        assert self.n_input_dims == working_space.get_extent('d')
        assert self.n_feature_maps == working_space.get_extent('f')

        X, working_space = working_space.transform(X, ['b', 'w', 'f', 'd'])

        X = X + self.b

        meta['space_above'] = working_space

        return X, meta, {}

    def bprop(self, delta, meta, fprop_state):
        meta['space_below'] = meta['space_above']
        return delta, meta

    def grads(self, delta, meta, fprop_state):
        working_space = meta['space_above']

        delta, working_space = working_space.transform(delta, ['f', 'd', 'bw'])
        grad_b = delta.sum(axis=2)

        return [grad_b]

    def params(self):
        return [self.b]

    def __repr__(self):
        return "{}(n_input_dims={}, n_feature_maps={})".format(
            self.__class__.__name__,
            self.n_input_dims,
            self.n_feature_maps)
