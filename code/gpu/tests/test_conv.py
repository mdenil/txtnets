__author__ = 'mdenil'

import numpy as np

import pycuda.autoinit
import pycuda
import pycuda.gpuarray

from gpu import conv


class TestFFTConvolve1D(object):
    def reference_convolve(self, X, K, mode):
        rows = []
        for x, k in zip(X, K):
            rows.append(
                np.convolve(x, k, mode=mode)
            )
        return np.vstack(rows)

    def check_allclose(self, actual, expected):
        assert np.allclose(actual, expected)

    def _run_fftconv1d(self, n_x, n_k, mode):
        n_rows = 2

        convolver = conv.FFTConv1D()

        X = np.random.standard_normal(size=(n_rows, n_x)).astype(np.float32)
        K = np.random.uniform(size=(n_rows, n_k)).astype(np.float32)

        actual = convolver.conv(pycuda.gpuarray.to_gpu(X), pycuda.gpuarray.to_gpu(K), mode=mode)
        expected = self.reference_convolve(X, K, mode=mode)

        print "X:", X.shape, "K:", K.shape, "A:", actual.shape, "E:", expected.shape
        print actual.get()
        print
        print expected
        print
        print "Max abs err:", np.max(np.abs(actual.get() - expected))

        assert np.max(np.abs(actual.get() - expected)) < 1e-5

    def test_fftconv1d(self):
        for n_x in [3, 10]:
            for n_k in [5, 6]:
                for mode in ['full', 'valid']:
                    # scipy doesn't accept this mode, and numpy does weird shit so I'm just going to not accept it
                    if mode == 'valid' and n_x < n_k:
                        continue
                    yield self._run_fftconv1d, n_x, n_k, mode
