__author__ = 'mdenil'

import numpy as np
import unittest

from gpu.utils import cpu_to_gpu

import cpu.optimize.objective.contrastive_multilingual
import gpu.optimize.objective.contrastive_multilingual


class TestGaussianEnergy(unittest.TestCase):
    def setUp(self):
        self.b = 100
        self.d = 25

        self.objective_cpu = cpu.optimize.objective.contrastive_multilingual.GaussianEnergy()
        self.x_cpu = np.random.standard_normal(size=(self.b, self.d))
        self.y_cpu = np.random.standard_normal(size=(self.b, self.d))

        self.objective_gpu = gpu.optimize.objective.contrastive_multilingual.GaussianEnergy()
        self.x_gpu = cpu_to_gpu(self.x_cpu.astype(np.float32))
        self.y_gpu = cpu_to_gpu(self.y_cpu.astype(np.float32))

    def test_fprop(self):
        actual = self.objective_gpu.fprop(self.x_gpu, self.y_gpu)
        expected = self.objective_cpu.fprop(self.x_cpu, self.y_cpu)

        self.assertTrue(np.allclose(actual.get(), expected, atol=1e-6))

    def test_bprop_x(self):
        c_cpu = np.random.standard_normal(size=(self.b, 1))
        c_gpu = cpu_to_gpu(c_cpu.astype(np.float32))

        actual, _ = self.objective_gpu.bprop(self.x_gpu, self.y_gpu, c_gpu)
        expected, _ = self.objective_cpu.bprop(self.x_cpu, self.y_cpu, c_cpu)

        self.assertTrue(np.allclose(actual.get(), expected, atol=1e-6))

    def test_bprop_y(self):
        c_cpu = np.random.standard_normal(size=(self.b, 1))
        c_gpu = cpu_to_gpu(c_cpu.astype(np.float32))

        _, actual = self.objective_gpu.bprop(self.x_gpu, self.y_gpu, c_gpu)
        _, expected = self.objective_cpu.bprop(self.x_cpu, self.y_cpu, c_cpu)

        self.assertTrue(np.allclose(actual.get(), expected, atol=1e-6))


class TestContrastiveHingeLoss(unittest.TestCase):
    def setUp(self):
        self.b = 100
        self.margin = 1.0

        self.objective_cpu = cpu.optimize.objective.contrastive_multilingual.ContrastiveHingeLoss(margin=self.margin)
        self.x_cpu = np.random.standard_normal(size=(self.b, 1)) ** 2
        self.y_cpu = np.random.standard_normal(size=(self.b, 1)) ** 2

        self.objective_gpu = gpu.optimize.objective.contrastive_multilingual.ContrastiveHingeLoss(margin=self.margin)
        self.x_gpu = cpu_to_gpu(self.x_cpu.astype(np.float32))
        self.y_gpu = cpu_to_gpu(self.y_cpu.astype(np.float32))

    def test_fprop(self):
        actual = self.objective_gpu.fprop(self.x_gpu, self.y_gpu)
        expected = self.objective_cpu.fprop(self.x_cpu, self.y_cpu)
        self.assertTrue(np.allclose(actual.get(), expected, atol=1e-6))

    def test_bprop_x(self):
        actual, _ = self.objective_gpu.bprop(self.x_gpu, self.y_gpu, 1.0)
        expected, _ = self.objective_cpu.bprop(self.x_cpu, self.y_cpu, 1.0)
        self.assertTrue(np.allclose(actual.get(), expected, atol=1e-6))

    def test_bprop_y(self):
        _, actual = self.objective_gpu.bprop(self.x_gpu, self.y_gpu, 1.0)
        _, expected = self.objective_cpu.bprop(self.x_cpu, self.y_cpu, 1.0)
        self.assertTrue(np.allclose(actual.get(), expected, atol=1e-6))