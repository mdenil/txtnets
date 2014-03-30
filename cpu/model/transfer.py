__author__ = 'mdenil'

import numpy as np
import pyfftw
import psutil

pyfftw.interfaces.cache.enable()

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

        self.W = 0.05 * np.random.standard_normal(size=(self.n_feature_maps * self.n_input_dimensions, self.kernel_width))

        self.input_axes = ['b', 'd', 'w']
        self.output_axes = ['b', 'f', 'd', 'w']

    def fprop(self, X, **meta):
        b, d, w = X.shape

        assert self.n_input_dimensions == d
        f = self.n_feature_maps

        X = np.reshape(
            np.transpose(
                np.concatenate([X[np.newaxis]] * f), # f b d w
                (1, 0, 2, 3)
            ),
            (b * f * d, w)
        )

        K = np.vstack([np.fliplr(self.W)] * b)
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
        meta['lengths'] = meta['lengths'] + self.kernel_width - 1

        X = np.reshape(
            X,
            (b, f, d, representation_length))

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
        self.input_axes = ['b', 'w', 'f', 'd']
        self.output_axes = ['b', 'w', 'f', 'd']

    def fprop(self, X, **meta):
        b, w, f, d = X.shape

        assert self.n_input_dims == d
        assert self.n_feature_maps == f

        X += self.b

        return X, meta

    def __repr__(self):
        return "{}(n_input_dims={}, n_feature_maps={})".format(
            self.__class__.__name__,
            self.n_input_dims,
            self.n_feature_maps)
