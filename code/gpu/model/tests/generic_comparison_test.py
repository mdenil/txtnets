__author__ = 'mdenil'

import numpy as np
import cPickle as pickle

import pycuda.gpuarray
import gpu.utils

import cpu.model.layer


class GenericCompareGPUToCPUTest(object):
    def test_fprop(self):
        Y_cpu, meta_cpu, fprop_state_cpu = self.layer_cpu.fprop(self.X_cpu, meta=dict(self.meta_cpu))
        Y_gpu, meta_gpu, fprop_state_gpu = self.layer_gpu.fprop(self.X_gpu, meta=dict(self.meta_gpu))

        self.assertLess(np.max(np.abs(Y_gpu.get() - Y_cpu.astype(np.float32))), 1e-5)

    def test_bprop(self):
        Y_cpu, meta_cpu, fprop_state_cpu = self.layer_cpu.fprop(self.X_cpu, meta=dict(self.meta_cpu))

        delta_cpu = np.random.standard_normal(size=Y_cpu.shape)
        delta_gpu = gpu.utils.cpu_to_gpu(delta_cpu.astype(np.float32))

        delta_cpu, meta_cpu = self.layer_cpu.bprop(delta_cpu, meta=dict(meta_cpu), fprop_state=fprop_state_cpu)
        delta_cpu, _ = meta_cpu['space_below'].transform(delta_cpu, self.X_cpu_space.axes)

        Y_gpu, meta_gpu, fprop_state_gpu = self.layer_gpu.fprop(self.X_gpu, meta=dict(self.meta_gpu))
        delta_gpu, meta_gpu = self.layer_gpu.bprop(delta_gpu, meta=dict(meta_gpu), fprop_state=fprop_state_gpu)
        delta_gpu, _ = meta_gpu['space_below'].transform(delta_gpu, self.X_gpu_space.axes)

        self.assertEqual(delta_gpu.shape, delta_cpu.shape)
        self.assertLess(np.max(np.abs(delta_gpu.get() - delta_cpu)), 1e-5)

    def test_grads(self):
        Y_cpu, meta_cpu, fprop_state_cpu = self.layer_cpu.fprop(self.X_cpu, meta=dict(self.meta_cpu))

        delta_cpu = np.random.standard_normal(size=Y_cpu.shape)
        delta_gpu = gpu.utils.cpu_to_gpu(delta_cpu.astype(np.float32))

        grads_cpu = self.layer_cpu.grads(delta_cpu, meta=dict(meta_cpu), fprop_state=fprop_state_cpu)

        Y_gpu, meta_gpu, fprop_state_gpu = self.layer_gpu.fprop(self.X_gpu, meta=dict(self.meta_gpu))
        grads_gpu = self.layer_gpu.grads(delta_gpu, meta=dict(meta_gpu), fprop_state=fprop_state_gpu)

        self.assertEqual(len(grads_cpu), len(grads_gpu))

        for g_cpu, g_gpu in zip(grads_cpu, grads_gpu):
            self.assertEqual(g_gpu.shape, g_cpu.shape)
            self.assertLess(np.max(np.abs(g_gpu.get() - g_cpu)), 1e-5)

    def test_pickles(self):
        layer_gpu_unpickled = pickle.loads(pickle.dumps(self.layer_gpu))

        self.assertEqual(len(layer_gpu_unpickled.params()), len(self.layer_gpu.params()))

        for p1, p2 in zip(layer_gpu_unpickled.params(), self.layer_gpu.params()):
            self.assertIsNot(p1, p2)
            self.assertIsInstance(p1, pycuda.gpuarray.GPUArray)
            self.assertEqual(type(p1), type(p2))
            self.assertTrue(np.all(p1.get() == p2.get()))

    def test_move_to_cpu(self):
        layer_gpu_to_cpu = self.layer_gpu.move_to_cpu()

        self.assertIsInstance(layer_gpu_to_cpu, cpu.model.layer.Layer)
        self.assertEqual(len(layer_gpu_to_cpu.params()), len(self.layer_gpu.params()))

        for pa, pe in zip(layer_gpu_to_cpu.params(), self.layer_gpu.params()):
            self.assertIsNot(pa, pe)
            self.assertIsInstance(pa, np.ndarray)
            self.assertTrue(np.all(pa == pe.get()))