__author__ = 'mdenil'

import numpy as np

from cpu import conv

# scipy.signal.fftconvolve

import unittest

class Convolution(unittest.TestCase):
    def setUp(self):
        self.X = np.random.standard_normal(size=(10, 100))
        self.K = np.random.uniform(size=(10, 6))

    def test_fftconv1d(self):
        actual = conv.fftconv1d(self.X, self.K)

        rows = []
        for x, k in zip(self.X, self.K):
            rows.append(
                np.convolve(x, k, mode='full')
            )

        expected = np.vstack(rows)

        assert np.allclose(actual, expected)