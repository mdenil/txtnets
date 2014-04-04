__author__ = 'mdenil'

import numpy as np
import pyfftw
import psutil

from cpu import space

pyfftw.interfaces.cache.enable()

class Softmax(object):
    def __init__(self,
                 n_classes,
                 n_input_dimensions):
        self.n_classes = n_classes
        self.n_input_dimensions = n_input_dimensions

        self.W = 0.05 * np.random.standard_normal(size=(self.n_classes, self.n_input_dimensions))
        self.b = np.zeros(shape=(self.n_classes,1))

    def fprop(self, X, **meta):
        input_space = meta['data_space']
        X, meta['data_space'] = input_space.transform(X, ['wfd', 'b'])

        X = np.exp(np.dot(self.W, X) + self.b)
        X /= np.sum(X, axis=0)

        meta['lengths'] = np.zeros(meta['lengths'].shape) + self.n_classes

        return X, meta

    def bprop(self, delta, **meta):
        out = np.dot(self.W.T, delta)
        return out, meta

    def grads(self, X, delta, **meta):
        X = self._flatten_axes(X)

        gw = np.dot(delta, X.T)
        gb = delta.sum(axis=1).reshape(self.b.shape)

        return [gw, gb], meta

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

    def fprop(self, X, **meta):
        data_space = meta['data_space']
        lengths = meta['lengths']

        b, d, w = data_space.get_extents(['b','d','w'])

        assert self.n_input_dimensions == d
        f = self.n_feature_maps

        X, data_space = data_space.transform(X, ['bfd', 'w'])
        X, data_space = data_space.broadcast(X, f=f)

        K, _ = self._kernel_space.broadcast(np.fliplr(self.W), b=b)
        kw = K.shape[1]

        # pad

        if w >= kw:
            p = int((w - kw) / 2)
            K = np.concatenate([
                np.zeros((K.shape[0], p)), K, np.zeros((K.shape[0], p))
            ], axis=1)
        else:
            p = int((kw - w) / 2)
            X = np.concatenate([
                np.zeros((X.shape[0], p)), X, np.zeros((X.shape[0], p))
            ], axis=1)

        # compute

        X = np.concatenate([X, np.zeros_like(X)], axis=1)
        K = np.concatenate([K, np.zeros_like(K)], axis=1)

        X = pyfftw.interfaces.numpy_fft.fft(X, axis=1, threads=self.n_threads)
        K = pyfftw.interfaces.numpy_fft.fft(K, axis=1, threads=self.n_threads)
        X = pyfftw.interfaces.numpy_fft.ifft(X*K, axis=1, threads=self.n_threads).real

        # trim

        X = X[:, p:-1-p]

        representation_length = X.shape[1]

        # length of a wide convolution
        lengths = lengths + self.kernel_width - 1

        data_space = data_space.set_extent(w=representation_length)

        meta['data_space'] = data_space
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

    def fprop(self, X, **meta):
        data_space = meta['data_space']

        assert [self.n_input_dims] == data_space.get_extents('d')
        assert [self.n_feature_maps] == data_space.get_extents('f')

        X, data_space = data_space.transform(X, ['b', 'w', 'f', 'd'])

        X += self.b

        meta['data_space'] = data_space

        return X, meta

    def __repr__(self):
        return "{}(n_input_dims={}, n_feature_maps={})".format(
            self.__class__.__name__,
            self.n_input_dims,
            self.n_feature_maps)
