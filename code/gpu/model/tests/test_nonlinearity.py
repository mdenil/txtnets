__author__ = 'mdenil'

import numpy as np
import unittest

from generic_comparison_test import GenericCompareGPUToCPUTest

import cpu.space
import cpu.model.nonlinearity

import gpu.utils
import gpu.space
import gpu.model.nonlinearity


class GenericNonlinearityTest(GenericCompareGPUToCPUTest):
    def setUp(self):
        w,f,d,b = 2, 3, 5, 10
        self.n_classes = 7

        self.X_cpu = np.random.standard_normal(size=(w, f, d, b))
        self.X_gpu = gpu.utils.cpu_to_gpu(self.X_cpu.astype(np.float32))

        self.X_cpu_space = cpu.space.CPUSpace.infer(self.X_cpu, ['w', 'f', 'd', 'b'])
        self.X_gpu_space = gpu.space.GPUSpace.infer(self.X_gpu, ['w', 'f', 'd', 'b'])

        self.Y_cpu = np.random.randint(0, self.n_classes, size=b)
        self.Y_cpu = np.equal.outer(self.Y_cpu, np.arange(self.n_classes)).astype(self.X_cpu.dtype)
        self.Y_gpu = gpu.utils.cpu_to_gpu(self.Y_cpu.astype(np.float32))

        self.layer_cpu = self.__class__.CPUModel()
        self.layer_gpu = self.__class__.GPUModel()
        
        self.meta_cpu = {'space_below': self.X_cpu_space, 'lengths': np.zeros(b) + w}
        self.meta_gpu = {'space_below': self.X_gpu_space, 'lengths': np.zeros(b) + w}


class Tanh(GenericNonlinearityTest, unittest.TestCase):
    CPUModel = cpu.model.nonlinearity.Tanh
    GPUModel = gpu.model.nonlinearity.Tanh


class Relu(GenericNonlinearityTest, unittest.TestCase):
    CPUModel = cpu.model.nonlinearity.Relu
    GPUModel = gpu.model.nonlinearity.Relu