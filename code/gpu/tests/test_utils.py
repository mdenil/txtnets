__author__ = 'mdenil'

import numpy as np
import unittest
import pycuda.autoinit

import gpu.utils
import cpu.space
import gpu.space

class TestFliplr(unittest.TestCase):
    def setUp(self):
        self.W = gpu.utils.cpu_to_gpu(np.random.standard_normal((500, 120)).astype(np.float32))

    def test_fliplr(self):
        actual = gpu.utils.fliplr(self.W)
        expected = np.fliplr(self.W.get())

        self.assertTrue(np.all(actual.get() == expected))


class SumAlongAxis(unittest.TestCase):
    def setUp(self):
        self.x_cpu = np.random.standard_normal(
            size=(2, 3, 4, 5, 6, 7, 8, 9)).astype(np.float32)
        self.x_cpu_space = cpu.space.CPUSpace.infer(
            self.x_cpu,
            ('2', '3', '4', '5', '6', '7', '8', '9'))

        self.x_gpu, self.x_gpu_space = gpu.space.GPUSpace.from_cpu(
            self.x_cpu, self.x_cpu_space)

    def test_sum_along_axis(self):
        for axis in self.x_gpu_space.folded_axes:
            actual, _ = gpu.utils.sum_along_axis(self.x_gpu, self.x_gpu_space, axis)
            expected = np.sum(self.x_cpu, self.x_cpu_space.axes.index(axis))

            self.assertTrue(np.all(actual.get() == expected))
