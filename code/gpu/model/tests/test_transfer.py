__author__ = 'mdenil'

import unittest
import numpy as np

import pycuda.gpuarray

import gpu.utils
import gpu.space
import gpu.model.cost
import gpu.model.transfer

import cpu.space
import cpu.model.cost
import cpu.model.transfer

np.random.seed(234234)


class Linear(unittest.TestCase):
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

        self.delta_mask_cpu = np.random.uniform(size=(b, 20)) > 0.5
        self.delta_mask_gpu = gpu.utils.cpu_to_gpu(self.delta_mask_cpu.astype(np.float32))


    def test_fprop(self):
        Y_cpu, meta_cpu, fprop_state_cpu = self.layer_cpu.fprop(self.X_cpu, meta=dict(self.meta_cpu))
        Y_gpu, meta_gpu, fprop_state_gpu = self.layer_gpu.fprop(self.X_gpu, meta=dict(self.meta_gpu))

        self.assertLess(np.max(np.abs(Y_gpu.get() - Y_cpu)), 1e-5)
        
    def test_bprop(self):
        Y_cpu, meta_cpu, fprop_state_cpu = self.layer_cpu.fprop(self.X_cpu, meta=dict(self.meta_cpu))
        delta_cpu, meta_cpu = self.layer_cpu.bprop(self.delta_mask_cpu, meta=dict(meta_cpu), fprop_state=fprop_state_cpu)
        delta_cpu, _ = meta_cpu['space_below'].transform(delta_cpu, self.X_cpu_space.axes)
        
        Y_gpu, meta_gpu, fprop_state_gpu = self.layer_gpu.fprop(self.X_gpu, meta=dict(self.meta_gpu))
        delta_gpu, meta_gpu = self.layer_gpu.bprop(self.delta_mask_gpu, meta=dict(meta_gpu), fprop_state=fprop_state_gpu)
        delta_gpu, _ = meta_gpu['space_below'].transform(delta_gpu, self.X_gpu_space.axes)

        self.assertLess(np.max(np.abs(delta_gpu.get() - delta_cpu)), 1e-5)

    def test_grad_W(self):
        Y_cpu, meta_cpu, fprop_state_cpu = self.layer_cpu.fprop(self.X_cpu, meta=dict(self.meta_cpu))
        [grad_W_cpu] = self.layer_cpu.grads(self.delta_mask_cpu, meta=dict(meta_cpu), fprop_state=fprop_state_cpu)

        Y_gpu, meta_gpu, fprop_state_gpu = self.layer_gpu.fprop(self.X_gpu, meta=dict(self.meta_gpu))
        [grad_W_gpu] = self.layer_gpu.grads(self.delta_mask_gpu, meta=dict(meta_gpu), fprop_state=fprop_state_gpu)

        self.assertLess(np.max(np.abs(grad_W_gpu.get() - grad_W_cpu)), 1e-5)


class Softmax(unittest.TestCase):
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

        self.X_space_cpu = cpu.space.CPUSpace.infer(self.X_cpu, ['w', 'f', 'd', 'b'])
        self.X_space_gpu = gpu.space.GPUSpace.infer(self.X_gpu, ['w', 'f', 'd', 'b'])

        self.meta_cpu = {'lengths': np.zeros(b) + w, 'space_below': self.X_space_cpu}
        self.meta_gpu = {'lengths': np.zeros(b) + w, 'space_below': self.X_space_gpu}

        self.cost_cpu = cpu.model.cost.CrossEntropy()
        self.cost_gpu = gpu.model.cost.CrossEntropy()

    def test_fprop(self):
        actual, _, _ = self.layer_gpu.fprop(self.X_gpu, meta=dict(self.meta_gpu))
        expected, _, _ = self.layer_cpu.fprop(self.X_cpu, meta=dict(self.meta_cpu))

        print np.max(np.abs(actual.get() - expected))

        self.assertLess(np.max(np.abs(actual.get() - expected)), 1e-5)

    def test_bprop(self):
        Y_cpu, meta_cpu, fprop_state_cpu = self.layer_cpu.fprop(self.X_cpu, meta=dict(self.meta_cpu))
        meta_cpu['space_below'] = meta_cpu['space_above']
        cost_cpu, meta_cpu, cost_state_cpu = self.cost_cpu.fprop(Y_cpu, self.Y_cpu, meta=dict(meta_cpu))
        delta_cpu, meta_cpu = self.cost_cpu.bprop(Y_cpu, self.Y_cpu, meta=dict(meta_cpu), fprop_state=cost_state_cpu)
        meta_cpu['space_above'] = meta_cpu['space_below']
        delta_cpu, meta_cpu = self.layer_cpu.bprop(delta_cpu, meta=dict(meta_cpu), fprop_state=fprop_state_cpu)
        delta_cpu, _ = meta_cpu['space_below'].transform(delta_cpu, self.X_space_cpu.axes)
        
        Y_gpu, meta_gpu, fprop_state_gpu = self.layer_gpu.fprop(self.X_gpu, meta=dict(self.meta_gpu))
        meta_gpu['space_below'] = meta_gpu['space_above']
        cost_gpu, meta_gpu, cost_state_gpu = self.cost_gpu.fprop(Y_gpu, self.Y_gpu, meta=dict(meta_gpu))
        delta_gpu, meta_gpu = self.cost_gpu.bprop(Y_gpu, self.Y_gpu, meta=dict(meta_gpu), fprop_state=cost_state_gpu)
        meta_gpu['space_above'] = meta_gpu['space_below']
        delta_gpu, meta_gpu = self.layer_gpu.bprop(delta_gpu, meta=dict(meta_gpu), fprop_state=fprop_state_gpu)
        delta_gpu, _ = meta_gpu['space_below'].transform(delta_gpu, self.X_space_gpu.axes)

        print np.max(np.abs(delta_gpu.get() - delta_cpu))

        self.assertLess(np.max(np.abs(delta_gpu.get() - delta_cpu)), 1e-5)

    def test_grad_W(self):
        Y_cpu, meta_cpu, fprop_state_cpu = self.layer_cpu.fprop(self.X_cpu, meta=dict(self.meta_cpu))
        meta_cpu['space_below'] = meta_cpu['space_above']
        cost_cpu, meta_cpu, cost_state_cpu = self.cost_cpu.fprop(Y_cpu, self.Y_cpu, meta=dict(meta_cpu))
        delta_cpu, meta_cpu = self.cost_cpu.bprop(Y_cpu, self.Y_cpu, meta=dict(meta_cpu), fprop_state=cost_state_cpu)
        meta_cpu['space_above'] = meta_cpu['space_below']
        [grad_W_cpu, _] = self.layer_cpu.grads(delta_cpu, meta=dict(meta_cpu), fprop_state=fprop_state_cpu)
        
        Y_gpu, meta_gpu, fprop_state_gpu = self.layer_gpu.fprop(self.X_gpu, meta=dict(self.meta_gpu))
        meta_gpu['space_below'] = meta_gpu['space_above']
        cost_gpu, meta_gpu, cost_state_gpu = self.cost_gpu.fprop(Y_gpu, self.Y_gpu, meta=dict(meta_gpu))
        delta_gpu, meta_gpu = self.cost_gpu.bprop(Y_gpu, self.Y_gpu, meta=dict(meta_gpu), fprop_state=cost_state_gpu)
        meta_gpu['space_above'] = meta_gpu['space_below']
        [grad_W_gpu, _] = self.layer_gpu.grads(delta_gpu, meta=dict(meta_gpu), fprop_state=fprop_state_gpu)

        self.assertLess(np.max(np.abs(grad_W_gpu.get() - grad_W_cpu)), 1e-5)

    def test_grad_b(self):
        Y_cpu, meta_cpu, fprop_state_cpu = self.layer_cpu.fprop(self.X_cpu, meta=dict(self.meta_cpu))
        meta_cpu['space_below'] = meta_cpu['space_above']
        cost_cpu, meta_cpu, cost_state_cpu = self.cost_cpu.fprop(Y_cpu, self.Y_cpu, meta=dict(meta_cpu))
        delta_cpu, meta_cpu = self.cost_cpu.bprop(Y_cpu, self.Y_cpu, meta=dict(meta_cpu), fprop_state=cost_state_cpu)
        meta_cpu['space_above'] = meta_cpu['space_below']
        [_, grad_b_cpu] = self.layer_cpu.grads(delta_cpu, meta=dict(meta_cpu), fprop_state=fprop_state_cpu)

        Y_gpu, meta_gpu, fprop_state_gpu = self.layer_gpu.fprop(self.X_gpu, meta=dict(self.meta_gpu))
        meta_gpu['space_below'] = meta_gpu['space_above']
        cost_gpu, meta_gpu, cost_state_gpu = self.cost_gpu.fprop(Y_gpu, self.Y_gpu, meta=dict(meta_gpu))
        delta_gpu, meta_gpu = self.cost_gpu.bprop(Y_gpu, self.Y_gpu, meta=dict(meta_gpu), fprop_state=cost_state_gpu)
        meta_gpu['space_above'] = meta_gpu['space_below']
        [_, grad_b_gpu] = self.layer_gpu.grads(delta_gpu, meta=dict(meta_gpu), fprop_state=fprop_state_gpu)

        self.assertLess(np.max(np.abs(grad_b_gpu.get() - grad_b_cpu)), 1e-5)



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

        self.assertLess(np.max(np.abs(Y_cpu - Y_gpu.get())), 1e-5)
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

        self.assertLess(np.max(np.abs(delta_cpu - delta_gpu.get())), 1e-5)

    def test_grad_W(self):
        Y_cpu, meta_cpu, fprop_state_cpu = self.layer_cpu.fprop(self.X_cpu, meta=dict(self.meta_cpu))
        Y_cpu *= self.delta_mask_cpu
        [grad_W_cpu] = self.layer_cpu.grads(self.delta_mask_cpu, meta=dict(meta_cpu), fprop_state=fprop_state_cpu)

        Y_gpu, meta_gpu, fprop_state_gpu = self.layer_gpu.fprop(self.X_gpu, meta=dict(self.meta_gpu))
        Y_gpu *= self.delta_mask_gpu
        [grad_W_gpu] = self.layer_gpu.grads(self.delta_mask_gpu, meta=dict(meta_gpu), fprop_state=fprop_state_gpu)

        self.assertLess(np.max(np.abs(grad_W_cpu - grad_W_gpu.get())), 1e-5)

class Bias(unittest.TestCase):
    def setUp(self):
        b,w,f,d = 2, 1, 3, 2

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


    def test_fprop(self):
        Y_cpu, meta_cpu, _ = self.layer_cpu.fprop(self.X_cpu, meta=dict(self.meta_cpu))
        Y_gpu, meta_gpu, _ = self.layer_gpu.fprop(self.X_gpu, meta=dict(self.meta_gpu))

        self.assertLess(np.max(np.abs(Y_cpu - Y_gpu.get())), 1e-5)


    def test_bprop(self):
        Y_cpu, meta_cpu, fprop_state_cpu = self.layer_cpu.fprop(self.X_cpu, meta=dict(self.meta_cpu))
        delta_cpu = np.ones_like(Y_cpu)
        delta_cpu, meta_cpu = self.layer_cpu.bprop(
            delta_cpu, meta=dict(meta_cpu),
            fprop_state=fprop_state_cpu)
        delta_cpu, _ = meta_cpu['space_below'].transform(delta_cpu, self.X_cpu_space.axes)

        Y_gpu, meta_gpu, fprop_state_gpu = self.layer_gpu.fprop(self.X_gpu, meta=dict(self.meta_gpu))
        delta_gpu = pycuda.gpuarray.zeros_like(Y_gpu) + 1.0
        delta_gpu, meta_gpu = self.layer_gpu.bprop(
            delta_gpu, meta=dict(meta_gpu),
            fprop_state=fprop_state_gpu)
        delta_gpu, _ = meta_gpu['space_below'].transform(delta_gpu, self.X_gpu_space.axes)

        self.assertLess(np.max(np.abs(delta_cpu - delta_gpu.get())), 1e-5)

    def test_grad_b(self):
        Y_cpu, meta_cpu, fprop_state_cpu = self.layer_cpu.fprop(self.X_cpu, meta=dict(self.meta_cpu))
        delta_cpu = np.ones_like(Y_cpu)
        [grad_b_cpu] = self.layer_cpu.grads(delta_cpu, meta=dict(meta_cpu), fprop_state=fprop_state_cpu)

        Y_gpu, meta_gpu, fprop_state_gpu = self.layer_gpu.fprop(self.X_gpu, meta=dict(self.meta_gpu))
        delta_gpu = pycuda.gpuarray.zeros_like(Y_gpu) + 1.0
        [grad_b_gpu] = self.layer_gpu.grads(delta_gpu, meta=dict(meta_gpu), fprop_state=fprop_state_gpu)

        self.assertLess(np.max(np.abs(grad_b_cpu - grad_b_gpu.get())), 1e-5)

    #
    # def test_grad_b(self):
    #     def func(b):
    #         self.layer.b = b.reshape(self.layer.b.shape)
    #         Y, meta, fprop_state = self.layer.fprop(self.X, meta=dict(self.meta))
    #         return Y.sum()
    #
    #     def grad(b):
    #         self.layer.b = b.reshape(self.layer.b.shape)
    #
    #         Y, meta, fprop_state = self.layer.fprop(self.X, meta=dict(self.meta))
    #         grads = self.layer.grads(np.ones_like(Y), meta=dict(meta), fprop_state=fprop_state)
    #
    #         gb = grads[0]
    #
    #         return gb.ravel()
    #
    #     assert scipy.optimize.check_grad(func, grad, self.layer.b.ravel()) < 1e-5
    #
