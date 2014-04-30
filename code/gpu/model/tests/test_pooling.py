__author__ = 'mdenil'

import numpy as np
import unittest

import cpu.space
import cpu.model.pooling

import gpu.utils
import gpu.space
import gpu.model.pooling


class Folding(object):
    def setUp(self):
        w, f, d, b = 3, 1, 20, 3

        self.X_cpu = np.random.standard_normal(size=(d, f, b, w))
        self.X_gpu = gpu.utils.cpu_to_gpu(self.X_cpu.astype(np.float32))

        self.X_cpu_space = cpu.space.CPUSpace.infer(self.X_cpu, ['d', 'f', 'b', 'w'])
        self.X_gpu_space = gpu.space.GPUSpace.infer(self.X_gpu, ['d', 'f', 'b', 'w'])

        self.layer_cpu = self.__class__.CPULayer()
        self.layer_gpu = self.__class__.GPULayer()

        self.meta_cpu = {'space_below': self.X_cpu_space,
                         'lengths': np.random.randint(low=1, high=w, size=b)}
        self.meta_gpu = {'space_below': self.X_gpu_space,
                         'lengths': np.random.randint(low=1, high=w, size=b)}


    def test_fprop(self):
        Y_cpu, _, _ = self.layer_cpu.fprop(self.X_cpu, meta=dict(self.meta_cpu))
        Y_gpu, _, _ = self.layer_gpu.fprop(self.X_gpu, meta=dict(self.meta_gpu))

        self.assertLess(np.max(np.abs(Y_gpu.get() - Y_cpu)), 1e-5)

    def test_bprop(self):
        Y_cpu, meta_cpu, fprop_state_cpu = self.layer_cpu.fprop(self.X_cpu, meta=dict(self.meta_cpu))
        delta = np.random.standard_normal(size=Y_cpu.shape)
        meta_cpu['space_below'] = meta_cpu['space_above']
        delta_cpu, _ = self.layer_cpu.bprop(delta, meta=dict(meta_cpu), fprop_state=fprop_state_cpu)

        Y_gpu, meta_gpu, fprop_state_gpu = self.layer_gpu.fprop(self.X_gpu, meta=dict(self.meta_gpu))
        delta = gpu.utils.cpu_to_gpu(delta.astype(np.float32))
        meta_gpu['space_below'] = meta_gpu['space_above']
        delta_gpu, _ = self.layer_gpu.bprop(delta, meta=dict(meta_gpu), fprop_state=fprop_state_gpu)

        self.assertLess(np.max(np.abs(delta_gpu.get() - delta_cpu)), 1e-5)


class SumFolding(Folding, unittest.TestCase):
    CPULayer = cpu.model.pooling.SumFolding
    GPULayer = gpu.model.pooling.SumFolding


class MaxFolding(Folding, unittest.TestCase):
    CPULayer = cpu.model.pooling.MaxFolding
    GPULayer = gpu.model.pooling.MaxFolding