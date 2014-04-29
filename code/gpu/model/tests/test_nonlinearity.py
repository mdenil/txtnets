__author__ = 'mdenil'

import numpy as np
import unittest

import cpu.space
import cpu.model.nonlinearity

import gpu.utils
import gpu.space
import gpu.model.nonlinearity

class NonlinearityTest(object):
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


class Tanh(NonlinearityTest, unittest.TestCase):
    CPUModel = cpu.model.nonlinearity.Tanh
    GPUModel = gpu.model.nonlinearity.Tanh


class Relu(NonlinearityTest, unittest.TestCase):
    CPUModel = cpu.model.nonlinearity.Relu
    GPUModel = gpu.model.nonlinearity.Relu