__author__ = 'mdenil'

import numpy as np
import unittest
import pycuda.autoinit

import gpu.utils

class TestFliplr(unittest.TestCase):
    def setUp(self):
        self.W = gpu.utils.cpu_to_gpu(np.random.standard_normal((500, 120)).astype(np.float32))

    def test_fliplr(self):
        actual = gpu.utils.fliplr(self.W)
        expected = np.fliplr(self.W.get())

        self.assertTrue(np.all(actual.get() == expected))
