__author__ = 'mdenil'

import numpy as np
import unittest

from generic_comparison_test import GenericCompareGPUToCPUTest

import cpu.space
import cpu.model.embedding

import gpu.utils
import gpu.space
import gpu.model.embedding


class WordEmbedding(GenericCompareGPUToCPUTest, unittest.TestCase):
    def setUp(self):
        d = 10
        # vocabulary size should be at least as big as the number of words to catch indexing errors
        vocabulary_size = 30

        self.layer_cpu = cpu.model.embedding.WordEmbedding(
            dimension=d,
            vocabulary_size=vocabulary_size,
            padding=0)

        self.layer_gpu = gpu.model.embedding.WordEmbedding(
            dimension=d,
            vocabulary_size=vocabulary_size,
            padding=0)
        self.layer_gpu.E = gpu.utils.cpu_to_gpu(self.layer_cpu.E.astype(np.float32))

        self.X_cpu = np.random.randint(vocabulary_size, size=(3, 5))
        self.X_gpu = gpu.utils.cpu_to_gpu(self.X_cpu.astype(np.int32))

        self.X_cpu_space = cpu.space.CPUSpace.infer(self.X_cpu, ['b', 'w'])
        self.X_gpu_space = gpu.space.GPUSpace.infer(self.X_gpu, ['b', 'w'])

        self.meta_cpu = {'lengths': np.zeros_like(self.X_cpu) + self.X_cpu.shape[1],
                         'space_below': self.X_cpu_space}
        self.meta_gpu = {'lengths': np.zeros_like(self.X_gpu) + self.X_gpu.shape[1],
                         'space_below': self.X_gpu_space}
