__author__ = 'mdenil'

import numpy as np
import psutil

from cpu import space
from cpu import conv

class Softmax(object):
    def __init__(self,
                 n_classes,
                 n_input_dimensions):
        self.n_classes = n_classes
        self.n_input_dimensions = n_input_dimensions

        self.W = 0.05 * np.random.standard_normal(size=(self.n_classes, self.n_input_dimensions))
        self.b = np.zeros(shape=(self.n_classes,1))

    def fprop(self, X, meta):
        working_space = meta['space_below']
        X, working_space = working_space.transform(X, ['wfd', 'b'])

        fprop_state = {'input_space': working_space }

        X = np.exp(np.dot(self.W, X) + self.b)
        X /= np.sum(X, axis=0)

        working_space = working_space.without_axes('wf')
        working_space = working_space.with_extent(d=self.n_classes)

        meta['lengths'] = np.zeros(meta['lengths'].shape) + self.n_classes
        meta['space_above'] = working_space

        return X, meta, fprop_state

    def bprop(self, Y, delta, meta, fprop_state):
        out = np.dot(self.W.T, delta)
        meta['space_below'] = fprop_state['input_space']
        return out, meta

    def grads(self, X, delta, meta, fprop_state):
        working_space = meta['space_below']
        X, working_space = working_space.transform(X, ['wfd', 'b'])

        grad_W = np.dot(delta, X.T)
        grad_b = delta.sum(axis=1).reshape(self.b.shape)
        return [grad_W, grad_b]

    def __repr__(self):
        return "{}(W={})".format(
            self.__class__.__name__,
            self.W.shape)

class SentenceConvolution(object):
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

        return X, meta

    def __repr__(self):
        return "{}(W={})".format(
            self.__class__.__name__,
            self.W.shape)


class Bias(object):
    def __init__(self, n_input_dims, n_feature_maps):
        self.n_input_dims = n_input_dims
        self.n_feature_maps = n_feature_maps

        self.b = np.zeros((n_feature_maps, n_input_dims))

    def fprop(self, X, meta):
        working_space = meta['space_below']

        assert [self.n_input_dims] == working_space.get_extents('d')
        assert [self.n_feature_maps] == working_space.get_extents('f')

        X, working_space = working_space.transform(X, ['b', 'w', 'f', 'd'])

        X = X + self.b

        meta['space_above'] = working_space

        return X, meta

    def bprop(self, Y, delta, meta, fprop_state):
        return delta, meta

    def grads(self, X, delta, meta):
        working_space = meta['space_above']

        delta, working_space = working_space.transform(delta, ['f', 'd', 'bw'])
        grad_b = delta.sum(axis=2)

        return [grad_b]

    def __repr__(self):
        return "{}(n_input_dims={}, n_feature_maps={})".format(
            self.__class__.__name__,
            self.n_input_dims,
            self.n_feature_maps)
