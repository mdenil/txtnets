__author__ = 'mdenil'

import numpy as np
import unittest

import cpu.space
import cpu.model.embedding

import gpu.utils
import gpu.space
import gpu.model.embedding

class WordEmbedding(unittest.TestCase):
    def setUp(self):
        d = 10
        # vocabulary size should be at least as big as the number of words to catch indexing errors
        vocabulary_size = 30

        self.layer_cpu = cpu.model.embedding.WordEmbedding(
            dimension=d,
            vocabulary_size=vocabulary_size)

        self.layer_gpu = gpu.model.embedding.WordEmbedding(
            dimension=d,
            vocabulary_size=vocabulary_size)
        self.layer_gpu.E = gpu.utils.cpu_to_gpu(self.layer_cpu.E.astype(np.float32))

        self.X_cpu = np.random.randint(vocabulary_size, size=(3, 5))
        self.X_gpu = gpu.utils.cpu_to_gpu(self.X_cpu)

        self.words_cpu_space = cpu.space.CPUSpace.infer(self.X_cpu, ['b', 'w'])
        self.words_gpu_space = gpu.space.GPUSpace.infer(self.X_gpu, ['b', 'w'])

        self.meta_cpu = {'lengths': np.zeros_like(self.X_cpu) + self.X_cpu.shape[1],
                         'space_below': self.words_cpu_space}
        self.meta_gpu = {'lengths': np.zeros_like(self.X_gpu) + self.X_gpu.shape[1],
                         'space_below': self.words_gpu_space}


    def test_fprop(self):
        Y_cpu, meta_cpu, fprop_state_cpu = self.layer_cpu.fprop(self.X_cpu, meta=dict(self.meta_cpu))
        Y_gpu, meta_gpu, fprop_state_gpu = self.layer_gpu.fprop(self.X_gpu, meta=dict(self.meta_gpu))

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
        
    def test_grads(self):
        Y_cpu, meta_cpu, fprop_state_cpu = self.layer_cpu.fprop(self.X_cpu, meta=dict(self.meta_cpu))
        delta_cpu = np.random.standard_normal(size=Y_cpu.shape)
        meta_cpu['space_below'] = meta_cpu['space_above']
        [grad_E_cpu] = self.layer_cpu.grads(delta_cpu, meta=dict(meta_cpu), fprop_state=fprop_state_cpu)
        
        Y_gpu, meta_gpu, fprop_state_gpu = self.layer_gpu.fprop(self.X_gpu, meta=dict(self.meta_gpu))
        delta_gpu = gpu.utils.cpu_to_gpu(delta_cpu.astype(np.float32))
        meta_gpu['space_below'] = meta_gpu['space_above']
        [grad_E_gpu] = self.layer_gpu.grads(delta_gpu, meta=dict(meta_gpu), fprop_state=fprop_state_gpu)

        self.assertLess(np.max(np.abs(grad_E_gpu.get() - grad_E_cpu)), 1e-5)
        