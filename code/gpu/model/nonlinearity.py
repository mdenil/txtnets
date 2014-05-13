__author__ = 'mdenil'

import numpy as np
import pycuda.compiler

import generic.model.nonlinearity

import gpu.model.layer

import jinja2

_nonlinearity_module_template = jinja2.Template("""
__global__ void fprop_kernel(float* X, float* out, int N)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N) {
        out[index] = {{ elementwise_fprop }};
    }
}

__global__ void bprop_kernel(float* delta, float* Y, float* out, int N)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N) {
        out[index] = {{ elementwise_bprop }};
    }
}
""")


class ElementwiseNonlinearity(gpu.model.layer.Layer):
    block_size = 512

    def __init__(self, *args, **kwargs):
        super(ElementwiseNonlinearity, self).__init__(*args, **kwargs)
        self.__acquire_device_kernels()

    def _fprop(self, X):
        # overwrites X
        self._fprop_kernel(
            X,
            X,
            np.int32(X.size),
            block=(self.__class__.block_size, 1, 1),
            grid=(X.size / self.__class__.block_size + 1, 1))

        return X

    def _bprop(self, delta, Y):
        # overwrites delta
        self._bprop_kernel(
            delta,
            Y,
            delta,
            np.int32(delta.size),
            block=(self.__class__.block_size, 1, 1),
            grid=(delta.size / self.__class__.block_size + 1, 1))

        return delta

    def __acquire_device_kernels(self):
        self._fprop_kernel = self.__class__._kernel_module.get_function("fprop_kernel")
        self._bprop_kernel = self.__class__._kernel_module.get_function("bprop_kernel")

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_fprop_kernel']
        del state['_bprop_kernel']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__acquire_device_kernels()

    def move_to_cpu(self):
        from gpu.model.host_device_component_mapping import get_cpu_analog
        cpu_class = get_cpu_analog(self.__class__)

        return cpu_class()


class Relu(generic.model.nonlinearity.Relu, ElementwiseNonlinearity):
    _kernel_module = pycuda.compiler.SourceModule(
        _nonlinearity_module_template.render(
            elementwise_fprop="fmax(0.0f, X[index])",
            elementwise_bprop="delta[index] * (float)(Y[index] > 0.0f)"))


class Tanh(generic.model.nonlinearity.Tanh, ElementwiseNonlinearity):
    _kernel_module = pycuda.compiler.SourceModule(
        _nonlinearity_module_template.render(
            elementwise_fprop="tanhf(X[index])",
            elementwise_bprop="delta[index] * (1.0f - Y[index]*Y[index])"))
