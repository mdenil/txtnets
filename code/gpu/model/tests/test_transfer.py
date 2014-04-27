__author__ = 'mdenil'

import unittest
import numpy as np

import gpu.utils
import gpu.space
import gpu.model.transfer

import cpu.space
import cpu.model.transfer

np.random.seed(234234)

class SentenceConvolution(unittest.TestCase):
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

        # Using this causes test_grad_W to fail if you forget to flip delta before the convolution when computing
        # the gradient (this is good because if you forget that you're doing it wrong).  If you don't have a mask and
        # just backprop all ones then the test still passes without the flip (i.e. with the wrong gradient).
        self.delta_mask_cpu = np.random.uniform(size=(b*d*f, w+kernel_width-1)) > 0.5
        self.delta_mask_gpu = gpu.utils.cpu_to_gpu(self.delta_mask_cpu.astype(np.float32))


    def test_fprop(self):
        Y_cpu, meta_cpu, _ = self.layer_cpu.fprop(self.X_cpu, meta=dict(self.meta_cpu))
        Y_gpu, meta_gpu, _ = self.layer_gpu.fprop(self.X_gpu, meta=dict(self.meta_gpu))

        self.assertTrue(np.allclose(Y_cpu, Y_gpu.get(), 1e-4))
        self.assertTrue(np.all(meta_cpu['lengths'] == meta_gpu['lengths']))

    def test_bprop(self):
        Y_cpu, meta_cpu, fprop_state_cpu = self.layer_cpu.fprop(self.X_cpu, meta=dict(self.meta_cpu))
        Y_cpu *= self.delta_mask_cpu
        delta_cpu, meta_cpu = self.layer_cpu.bprop(
            self.delta_mask_cpu, meta=dict(meta_cpu),
            fprop_state=fprop_state_cpu)
        delta_cpu, _ = meta_cpu['space_below'].transform(delta_cpu, self.X_cpu_space.axes)

        Y_gpu, meta_gpu, fprop_state_gpu = self.layer_gpu.fprop(self.X_gpu, meta=dict(self.meta_gpu))
        Y_gpu *= self.delta_mask_gpu
        delta_gpu, meta_gpu = self.layer_gpu.bprop(
            self.delta_mask_gpu, meta=dict(meta_gpu),
            fprop_state=fprop_state_gpu)
        delta_gpu, _ = meta_gpu['space_below'].transform(delta_gpu, self.X_gpu_space.axes)

        self.assertTrue(np.allclose(delta_cpu, delta_gpu.get(), 1e-4))

    def test_grad_W(self):
        Y_cpu, meta_cpu, fprop_state_cpu = self.layer_cpu.fprop(self.X_cpu, meta=dict(self.meta_cpu))
        Y_cpu *= self.delta_mask_cpu
        [grad_W_cpu] = self.layer_cpu.grads(self.delta_mask_cpu, meta=dict(meta_cpu), fprop_state=fprop_state_cpu)

        Y_gpu, meta_gpu, fprop_state_gpu = self.layer_gpu.fprop(self.X_gpu, meta=dict(self.meta_gpu))
        Y_gpu *= self.delta_mask_gpu
        [grad_W_gpu] = self.layer_gpu.grads(self.delta_mask_gpu, meta=dict(meta_gpu), fprop_state=fprop_state_gpu)

        self.assertTrue(np.allclose(grad_W_cpu, grad_W_gpu.get(), 1e-4))

    # def test_grad_W(self):
    #     def func(W):
    #         self.layer.W = W.reshape(self.layer.W.shape)
    #         Y, meta, fprop_state = self.layer.fprop(self.X.copy(), meta=dict(self.meta))
    #         Y *= self.delta_mask
    #         return Y.sum()
    #
    #     def grad(W):
    #         self.layer.W = W.reshape(self.layer.W.shape)
    #
    #         Y, meta, fprop_state = self.layer.fprop(self.X.copy(), meta=dict(self.meta))
    #         delta = np.ones_like(Y)
    #         [grad_W] = self.layer.grads(self.delta_mask, meta=dict(meta), fprop_state=fprop_state)
    #
    #         return grad_W.ravel()
    #
    #     assert scipy.optimize.check_grad(func, grad, self.layer.W.ravel()) < 1e-5
