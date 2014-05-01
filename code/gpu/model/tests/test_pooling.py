__author__ = 'mdenil'

import numpy as np
import unittest

from generic_comparison_test import GenericCompareGPUToCPUTest

import cpu.space
import cpu.model.pooling

import gpu.utils
import gpu.space
import gpu.model.pooling


class KMaxPooling(GenericCompareGPUToCPUTest, unittest.TestCase):
    def setUp(self):
        w,f,d,b = 10, 2, 3, 5
        self.k = 4

        self.X_cpu = np.random.standard_normal(size=(d, f, b, w))
        self.X_gpu = gpu.utils.cpu_to_gpu(self.X_cpu.astype(np.float32))

        self.X_cpu_space = cpu.space.CPUSpace.infer(self.X_cpu, ['d', 'f', 'b', 'w'])
        self.X_gpu_space = gpu.space.GPUSpace.infer(self.X_gpu, ['d', 'f', 'b', 'w'])

        self.layer_cpu = cpu.model.pooling.KMaxPooling(self.k)
        self.layer_gpu = gpu.model.pooling.KMaxPooling(self.k)

        self.meta_cpu = {
            'space_below': self.X_cpu_space,
            'lengths': np.random.randint(low=1, high=w, size=b)
        }
        self.meta_gpu = {
            'space_below': self.X_gpu_space,
            'lengths': self.meta_cpu['lengths'].copy()
        }


class DynamicKMaxPooling(GenericCompareGPUToCPUTest, unittest.TestCase):
    def setUp(self):
        self.k = 4
        self.k_dynamic = 0.4

        self.X_cpu = np.array(
            [
                [5, 3, 1, 6, 2, 1],
                [2, 0, 1, 0, 8, 9],
                [1, 7, 1, 8, 2, 9],
            ]).astype(np.float)
        self.X_gpu = gpu.utils.cpu_to_gpu(self.X_cpu.astype(np.float32))

        self.X_cpu_space = cpu.space.CPUSpace.infer(self.X_cpu, ['b', 'w'])
        self.X_gpu_space = gpu.space.GPUSpace.infer(self.X_gpu, ['b', 'w'])

        lengths = np.array([4, 6, 2])

        self.layer_cpu = cpu.model.pooling.KMaxPooling(k=self.k, k_dynamic=self.k_dynamic)
        self.layer_gpu = gpu.model.pooling.KMaxPooling(k=self.k, k_dynamic=self.k_dynamic)

        self.meta_cpu = {
            'space_below': self.X_cpu_space,
            'lengths': lengths.copy(),
        }
        self.meta_gpu = {
            'space_below': self.X_gpu_space,
            'lengths': lengths.copy(),
        }


class GenericFoldingTest(GenericCompareGPUToCPUTest):
    def setUp(self):
        w, f, d, b = 3, 1, 20, 3

        self.X_cpu = np.random.standard_normal(size=(d, f, b, w))
        self.X_gpu = gpu.utils.cpu_to_gpu(self.X_cpu.astype(np.float32))

        self.X_cpu_space = cpu.space.CPUSpace.infer(self.X_cpu, ['d', 'f', 'b', 'w'])
        self.X_gpu_space = gpu.space.GPUSpace.infer(self.X_gpu, ['d', 'f', 'b', 'w'])

        self.layer_cpu = self.__class__.CPULayer()
        self.layer_gpu = self.__class__.GPULayer()

        self.meta_cpu = {
            'space_below': self.X_cpu_space,
            'lengths': np.random.randint(low=1, high=w, size=b)
        }
        self.meta_gpu = {
            'space_below': self.X_gpu_space,
            'lengths': np.random.randint(low=1, high=w, size=b)
        }


class SumFolding(GenericFoldingTest, unittest.TestCase):
    CPULayer = cpu.model.pooling.SumFolding
    GPULayer = gpu.model.pooling.SumFolding


class MaxFolding(GenericFoldingTest, unittest.TestCase):
    CPULayer = cpu.model.pooling.MaxFolding
    GPULayer = gpu.model.pooling.MaxFolding