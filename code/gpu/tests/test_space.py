__author__ = 'mdenil'

import itertools

import pycuda

import numpy as np
import unittest

from gpu.space import GPUSpace
from cpu.space import CPUSpace

import gpu.utils


class TestGPUSpace(unittest.TestCase):
    def setUp(self):
        self.X_cpu = np.zeros((3,5,4)).astype(np.float32)
        self.cpu_space = CPUSpace.infer(self.X_cpu, ('a', 'b', 'c'))

    def test_from_cpu(self):
        X_gpu, gpu_space = GPUSpace.from_cpu(self.X_cpu, self.cpu_space)

        self.assertTrue(isinstance(X_gpu, pycuda.gpuarray.GPUArray))
        self.assertTrue(np.allclose(gpu.utils.gpu_to_cpu(X_gpu), self.X_cpu))
        self.assertEqual(gpu_space.axes, self.cpu_space.axes)
        self.assertEqual(gpu_space.extents, self.cpu_space.extents)

    def test_to_cpu(self):
        X_gpu, gpu_space = GPUSpace.from_cpu(self.X_cpu, self.cpu_space)
        X_cpu, cpu_space = gpu_space.to_cpu(X_gpu)

        self.assertTrue(isinstance(X_cpu, np.ndarray))
        self.assertTrue(np.allclose(X_cpu, self.X_cpu))
        self.assertEqual(cpu_space.axes, gpu_space.axes)
        self.assertEqual(cpu_space.extents, gpu_space.extents)

    def test_transpose(self):
        X_gpu, gpu_space = GPUSpace.from_cpu(self.X_cpu, self.cpu_space)

        for axes in itertools.permutations(self.cpu_space.axes):
            Y_cpu, Y_cpu_space = self.cpu_space.transpose(self.X_cpu, axes)
            Y_gpu, Y_gpu_space = gpu_space.transpose(X_gpu, axes)

            self.assertTrue(np.allclose(gpu.utils.gpu_to_cpu(Y_gpu), Y_cpu))
            self.assertEqual(Y_gpu_space.axes, Y_cpu_space.axes)
            self.assertEqual(Y_gpu_space.extents, Y_cpu_space.extents)

    def test_broadcast(self):
        X_gpu, gpu_space = GPUSpace.from_cpu(self.X_cpu, self.cpu_space)

        # broadcast along a new dimension
        Y_cpu, Y_cpu_space = self.cpu_space.broadcast(self.X_cpu, f=3)
        Y_gpu, Y_gpu_space = gpu_space.broadcast(X_gpu, f=3)

        self.assertTrue(np.allclose(gpu.utils.gpu_to_cpu(Y_gpu), Y_cpu))
        self.assertEqual(Y_gpu_space.axes, Y_cpu_space.axes)
        self.assertEqual(Y_gpu_space.extents, Y_cpu_space.extents)

        # broadcast along an existing dimension
        Y_cpu, Y_cpu_space = self.cpu_space.broadcast(self.X_cpu, b=3)
        Y_gpu, Y_gpu_space = gpu_space.broadcast(X_gpu, b=3)

        self.assertTrue(np.allclose(gpu.utils.gpu_to_cpu(Y_gpu), Y_cpu))
        self.assertEqual(Y_gpu_space.axes, Y_cpu_space.axes)
        self.assertEqual(Y_gpu_space.extents, Y_cpu_space.extents)

        # broadcast along multiple existing dimensions
        Y_cpu, Y_cpu_space = self.cpu_space.broadcast(self.X_cpu, b=3, a=4)
        Y_gpu, Y_gpu_space = gpu_space.broadcast(X_gpu, b=3, a=4)

        self.assertTrue(np.allclose(gpu.utils.gpu_to_cpu(Y_gpu), Y_cpu))
        self.assertEqual(Y_gpu_space.axes, Y_cpu_space.axes)
        self.assertEqual(Y_gpu_space.extents, Y_cpu_space.extents)

        # broadcast along multiple new dimensions
        Y_cpu, Y_cpu_space = self.cpu_space.broadcast(self.X_cpu, f=3, g=4)
        Y_gpu, Y_gpu_space = gpu_space.broadcast(X_gpu, f=3, g=4)

        self.assertTrue(np.allclose(gpu.utils.gpu_to_cpu(Y_gpu), Y_cpu))
        self.assertEqual(Y_gpu_space.axes, Y_cpu_space.axes)
        self.assertEqual(Y_gpu_space.extents, Y_cpu_space.extents)

        # broadcast along multiple dimensions, one new one old
        Y_cpu, Y_cpu_space = self.cpu_space.broadcast(self.X_cpu, a=3, g=4)
        Y_gpu, Y_gpu_space = gpu_space.broadcast(X_gpu, a=3, g=4)

        self.assertTrue(np.allclose(gpu.utils.gpu_to_cpu(Y_gpu), Y_cpu))
        self.assertEqual(Y_gpu_space.axes, Y_cpu_space.axes)
        self.assertEqual(Y_gpu_space.extents, Y_cpu_space.extents)