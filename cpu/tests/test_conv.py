__author__ = 'mdenil'

import numpy as np

from cpu import conv

# import unittest

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

        X = np.random.standard_normal(size=(n_rows, n_x))
        K = np.random.uniform(size=(n_rows, n_k))

        actual = conv.fftconv1d(X, K, mode=mode)
        expected = self.reference_convolve(X, K, mode=mode)

        # print actual.shape, expected.shape
        # print actual
        # print expected
        assert np.allclose(actual, expected)

    def test_fftconv1d(self):
        n_rows = 2
        for n_x in [10]: #[4, 10]:
            for n_k in [5, 6]:
                for mode in ['full', 'valid']:
                    yield self._run_fftconv1d, n_x, n_k, mode
