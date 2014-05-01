__author__ = 'mdenil'

import unittest
import numpy as np

from generic_comparison_test import GenericCompareGPUToCPUTest

import gpu.utils
import gpu.space
import gpu.model.cost
import gpu.model.transfer

import cpu.space
import cpu.model.cost
import cpu.model.transfer

np.random.seed(234234)


class Linear(GenericCompareGPUToCPUTest, unittest.TestCase):
    def setUp(self):
        b,w,f,d = 2, 20, 2, 2

        self.layer_cpu = cpu.model.transfer.Linear(
            n_input=f*d*w,
            n_output=20)
        
        self.layer_gpu = gpu.model.transfer.Linear(
            n_input=f*d*w,
            n_output=20)
        
        self.layer_cpu.W = self.layer_gpu.W.get()

        self.X_cpu = np.random.standard_normal(size=(b,w,d,f))
        self.X_gpu = gpu.utils.cpu_to_gpu(self.X_cpu.astype(np.float32))

        self.X_cpu_space = cpu.space.CPUSpace.infer(self.X_cpu, ['b', 'w', 'd', 'f'])
        self.X_gpu_space = gpu.space.GPUSpace.infer(self.X_gpu, ['b', 'w', 'd', 'f'])

        self.meta_cpu = {'lengths': np.random.randint(1, w, size=b), 'space_below': self.X_cpu_space}
        self.meta_gpu = {'lengths': np.random.randint(1, w, size=b), 'space_below': self.X_gpu_space}


class Softmax(GenericCompareGPUToCPUTest, unittest.TestCase):
    def setUp(self):
        # X = ['w', 'f', 'd', 'b']
        # Y = ['d', 'b'] (d = classes)
        w,f,d,b = 2, 3, 5, 10
        self.n_input_dimensions = w*f*d
        self.n_classes = 7

        self.layer_cpu = cpu.model.transfer.Softmax(
            n_classes=self.n_classes,
            n_input_dimensions=self.n_input_dimensions)

        self.layer_gpu = gpu.model.transfer.Softmax(
            n_classes=self.n_classes,
            n_input_dimensions=self.n_input_dimensions)
        self.layer_cpu.W = self.layer_gpu.W.get()
        self.layer_cpu.b = self.layer_gpu.b.get()

        self.X_cpu = np.random.standard_normal(size=(w, f, d, b))
        self.Y_cpu = np.random.randint(0, self.n_classes, size=b)
        self.Y_cpu = np.equal.outer(self.Y_cpu, np.arange(self.n_classes)).astype(self.X_cpu.dtype)

        self.X_gpu = gpu.utils.cpu_to_gpu(self.X_cpu.astype(np.float32))
        self.Y_gpu = gpu.utils.cpu_to_gpu(self.Y_cpu.astype(np.float32))

        self.X_cpu_space = cpu.space.CPUSpace.infer(self.X_cpu, ['w', 'f', 'd', 'b'])
        self.X_gpu_space = gpu.space.GPUSpace.infer(self.X_gpu, ['w', 'f', 'd', 'b'])

        self.meta_cpu = {'lengths': np.zeros(b) + w, 'space_below': self.X_cpu_space}
        self.meta_gpu = {'lengths': np.zeros(b) + w, 'space_below': self.X_gpu_space}


class SentenceConvolution(GenericCompareGPUToCPUTest, unittest.TestCase):
    def setUp(self):
        b,w,f,d,c = 2, 20, 2, 2, 2
        kernel_width = 4

        self.layer_cpu = cpu.model.transfer.SentenceConvolution(
            n_feature_maps=f,
            n_input_dimensions=d,
            n_channels=c,
            kernel_width=kernel_width)

        self.layer_gpu = gpu.model.transfer.SentenceConvolution(
            n_feature_maps=f,
            n_input_dimensions=d,
            n_channels=c,
            kernel_width=kernel_width)

        # Set the GPU parameters to be equal to the CPU parameters
        self.layer_gpu.W.set(self.layer_cpu.W.astype(np.float32))

        self.X_cpu = np.random.standard_normal(size=(b,w,d,c))
        self.X_gpu = gpu.utils.cpu_to_gpu(self.X_cpu.astype(np.float32))

        self.X_cpu_space = cpu.space.CPUSpace.infer(self.X_cpu, ['b', 'w', 'd', 'f'])
        self.meta_cpu = {
            'lengths': np.random.randint(1, w, size=b),
            'space_below': self.X_cpu_space,
            }

        self.X_gpu_space = gpu.space.GPUSpace.infer(self.X_gpu, ['b', 'w', 'd', 'f'])
        self.meta_gpu = {
            'lengths': self.meta_cpu['lengths'],
            'space_below': self.X_gpu_space,
            }


class Bias(GenericCompareGPUToCPUTest, unittest.TestCase):
    def setUp(self):
        b, w, f, d = 2, 1, 3, 2

        self.layer_cpu = cpu.model.transfer.Bias(
            n_feature_maps=f,
            n_input_dims=d)
        # biases default to zero, lets mix it up a bit
        self.layer_cpu.b = np.random.standard_normal(size=self.layer_cpu.b.shape)

        self.layer_gpu = gpu.model.transfer.Bias(
            n_feature_maps=f,
            n_input_dims=d)
        self.layer_gpu.b = gpu.utils.cpu_to_gpu(self.layer_cpu.b.astype(np.float32))

        self.X_cpu = np.random.standard_normal(size=(b,w,f,d))
        self.X_gpu = gpu.utils.cpu_to_gpu(self.X_cpu.astype(np.float32))

        self.X_cpu_space = cpu.space.CPUSpace.infer(self.X_cpu, ('b', 'w', 'f', 'd'))
        self.X_gpu_space = gpu.space.GPUSpace.infer(self.X_gpu, ('b', 'w', 'f', 'd'))

        self.meta_cpu = {'lengths': np.zeros(b) + w, 'space_below': self.X_cpu_space}
        self.meta_gpu = {'lengths': np.zeros(b) + w, 'space_below': self.X_gpu_space}
