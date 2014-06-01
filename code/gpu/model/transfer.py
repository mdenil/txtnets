__author__ = 'mdenil'

import numpy as np

import pycuda.autoinit
from pycuda import cumath

import gpu.utils
from gpu import space
from gpu.model import layer

import generic.model.transfer

import gpu.conv

import scikits.cuda.linalg
scikits.cuda.linalg.init()


class Linear(generic.model.transfer.Linear, layer.Layer):
    def __init__(self, *args, **kwargs):
        super(Linear, self).__init__(*args, **kwargs)
        self.W = gpu.utils.cpu_to_gpu(self.W.astype(np.float32))

    def _fprop(self, X):
        Y = scikits.cuda.linalg.dot(X, self.W)
        Y_space = space.GPUSpace.infer(Y, ('b', 'd'))
        return Y, Y_space

    def _bprop(self, delta):
        return scikits.cuda.linalg.dot(delta, self.W, transb='T')

    def _grads(self, X, delta):
        return [scikits.cuda.linalg.dot(X, delta, transa='T')]

    def move_to_cpu(self):
        from gpu.model.host_device_component_mapping import get_cpu_analog
        cpu_class = get_cpu_analog(self.__class__)

        return cpu_class(
            n_input=self.n_input,
            n_output=self.n_output,
            W=gpu.utils.gpu_to_cpu(self.W))


class Softmax(generic.model.transfer.Softmax, layer.Layer):
    def __init__(self, *args, **kwargs):
        super(Softmax, self).__init__(*args, **kwargs)

        self.W = gpu.utils.cpu_to_gpu(self.W.astype(np.float32))
        self.b = gpu.utils.cpu_to_gpu(self.b.astype(np.float32))
        self._b_space = space.GPUSpace.infer(self.b, ('b', 'w'))
        self._sum_vector_classes = gpu.utils.cpu_to_gpu(np.ones((self.n_classes, 1), dtype=np.float32))

    def _fprop(self, X, X_space):
        A = scikits.cuda.linalg.dot(X, self.W)
        B, bias_space = self._b_space.broadcast(self.b, b=X_space.get_extent('b'))
        Y = cumath.exp(A + B)

        Z = scikits.cuda.linalg.dot(Y, self._sum_vector_classes)
        Z_space = bias_space.with_extents(w=1)
        Z, Z_space = Z_space.broadcast(Z, w=self.n_classes)

        Y /= Z

        return Y

    def _bprop(self, delta, Y):
        return scikits.cuda.linalg.dot(delta * Y * (1.0 - Y), self.W, transb='T')

    def _grads(self, delta, X, Y):
        delta *= Y * (1.0 - Y)
        grad_W = scikits.cuda.linalg.dot(X, delta, transa='T')

        # FIXME: gpu.utils.sum_along_axis
        sum_vector_batch = pycuda.gpuarray.zeros((delta.shape[0], 1), dtype=np.float32)
        sum_vector_batch += 1.0

        grad_b = scikits.cuda.linalg.dot(delta, sum_vector_batch, transa='T').reshape(self.b.shape)

        return [grad_W, grad_b]

    def move_to_cpu(self):
        from gpu.model.host_device_component_mapping import get_cpu_analog
        cpu_class = get_cpu_analog(self.__class__)

        return cpu_class(
            n_classes=self.n_classes,
            n_input_dimensions=self.n_input_dimensions,
            W=gpu.utils.gpu_to_cpu(self.W),
            b=gpu.utils.gpu_to_cpu(self.b))


class SentenceConvolution(generic.model.transfer.SentenceConvolution, layer.Layer):
    def __init__(self, *args, **kwargs):
        super(SentenceConvolution, self).__init__(*args, **kwargs)

        self.W, self._kernel_space = gpu.space.GPUSpace.from_cpu(
            self.W.astype(np.float32),
            self._kernel_space)

        self._conv = gpu.conv.FFTConv1D()

    def _fprop(self, X, X_space):
        K, _ = self._kernel_space.broadcast(gpu.utils.fliplr(self.W), b=X_space.get_extent('b'))
        X = self._conv.conv(X, K)

        X_space = X_space.with_extents(w=X.shape[1])

        X, X_space = gpu.utils.sum_along_axis(X, X_space, 'c')
        X, X_space = X_space.transform(X, (('b', 'd', 'f'), 'w'))

        return X, X_space

    def _bprop(self, delta, delta_space):
        K, _ = self._kernel_space.broadcast(self.W, b=delta_space.get_extent('b'))

        delta = self._conv.conv(delta, K, mode='valid')
        delta_space = delta_space.with_extents(w=delta.shape[1])

        delta, delta_space = gpu.utils.sum_along_axis(delta, delta_space, 'f')
        delta_space = delta_space.rename_axes(c='f')

        return delta, delta_space

    def _grads(self, delta, delta_space, X):
        grad_W = self._conv.conv(gpu.utils.fliplr(delta), X, mode='valid')
        grad_W_space = delta_space.with_extents(w=grad_W.shape[1])

        grad_W, grad_W_space = gpu.utils.sum_along_axis(grad_W, grad_W_space, 'b')
        grad_W, grad_W_space = grad_W_space.transform(grad_W, [('b', 'd', 'f', 'c'), 'w'])

        return [grad_W]

    def move_to_cpu(self):
        from gpu.model.host_device_component_mapping import get_cpu_analog
        cpu_class = get_cpu_analog(self.__class__)

        return cpu_class(
            n_feature_maps=self.n_feature_maps,
            kernel_width=self.kernel_width,
            n_input_dimensions=self.n_input_dimensions,
            n_channels=self.n_channels,
            W=gpu.utils.gpu_to_cpu(self.W))


class Bias(generic.model.transfer.Bias, layer.Layer):
    def __init__(self, *args, **kwargs):
        super(Bias, self).__init__(*args, **kwargs)

        self.b = gpu.utils.cpu_to_gpu(self.b.astype(np.float32))
        self._b_space = space.GPUSpace.infer(self.b, ('d', 'f'))

    def _fprop(self, X, X_space):
        B, _ = self._b_space.transform(
            self.b,
            X_space.axes,
            w=X_space.get_extent('w'),
            b=X_space.get_extent('b'))

        X += B
        return X

    # bprop is a no-op

    def _grads(self, delta, delta_space):
        delta, delta_space = gpu.utils.sum_along_axis(delta, delta_space, 'b')
        grad_b, grad_b_space = gpu.utils.sum_along_axis(delta, delta_space, 'w')
        grad_b, grad_b_space = grad_b_space.transform(grad_b, self._b_space.axes)
        return [grad_b]

    def move_to_cpu(self):
        from gpu.model.host_device_component_mapping import get_cpu_analog
        cpu_class = get_cpu_analog(self.__class__)

        return cpu_class(
            n_input_dims=self.n_input_dims,
            n_feature_maps=self.n_feature_maps,
            b=gpu.utils.gpu_to_cpu(self.b))


class AxisReduction(generic.model.transfer.AxisReduction, layer.Layer):
    def _fprop(self, X, X_space):
        X, X_space = gpu.utils.sum_along_axis(X, X_space, self.axis)
        return X, X_space

    # bprop is generic
    # no grads

    def move_to_cpu(self):
        from gpu.model.host_device_component_mapping import get_cpu_analog
        cpu_class = get_cpu_analog(self.__class__)

        return cpu_class(axis=self.axis)


class ReshapeForDocuments(generic.model.transfer.ReshapeForDocuments, layer.Layer):
    Space = space.GPUSpace

    def move_to_cpu(self):
        from gpu.model.host_device_component_mapping import get_cpu_analog
        cpu_class = get_cpu_analog(self.__class__)

        return cpu_class()