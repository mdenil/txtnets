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
    block_size = 256

    def __init__(self):
        self._fprop_kernel = self._kernel_module.get_function("fprop_kernel")
        self._bprop_kernel = self._kernel_module.get_function("bprop_kernel")

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
