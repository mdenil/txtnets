__author__ = 'mdenil'

import numpy as np

import generic.model.pooling

import gpu.model.layer

import pycuda.gpuarray
import pycuda.compiler

# X should be size (N, M)
# out should be size (N/2, M)
# start this kernel with a 2d group of threads of shape (N/2, M)
# N should be even
_sum_folding_module = pycuda.compiler.SourceModule("""
__global__ void fprop_kernel(float* X, int N, int M, float* out)
{
    const int r1 = blockIdx.x * blockDim.x + threadIdx.x;
    const int r2 = r1 + N/2;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;

    if (r1 < N/2 && c < M) {
        out[r1 * M + c] = X[r1 * M + c] + X[r2 * M + c];
    }
}
""")


class SumFolding(generic.model.pooling.SumFolding, gpu.model.layer.Layer):
    block_size = 256

    def __init__(self):
        self._fprop_kernel = _sum_folding_module.get_function("fprop_kernel")

    def _fprop(self, X):
        out = pycuda.gpuarray.empty((X.shape[0]//2, X.shape[1]), dtype=np.float32)

        rows_per_block = self.__class__.block_size // X.shape[1] + 1
        num_blocks = X.shape[0] // rows_per_block + 1

        self._fprop_kernel(
            X,
            np.int32(X.shape[0]),
            np.int32(X.shape[1]),
            out,
            block=(rows_per_block, X.shape[1], 1),
            grid=(num_blocks, 1))

        return out

    # bprop is completely generic
    # there are no grads


# X should be size (N, M)
# out should be size (N/2, M)
# switches should be size (N, M)
# start this kernel with a 2d group of threads of shape (N/2, M)
# N should be even
_max_folding_module = pycuda.compiler.SourceModule("""
__global__ void fprop_kernel(float* X, int N, int M, float* out, float* switches)
{
    const int r1 = blockIdx.x * blockDim.x + threadIdx.x;
    const int r2 = r1 + N/2;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;

    if (r1 < N/2 && c < M) {
        const float v1 = X[r1 * M + c];
        const float v2 = X[r2 * M + c];

        out[r1 * M + c] = fmaxf(v1, v2);
        switches[r1 * M + c] = (float)(v1 >  v2);
        switches[r2 * M + c] = (float)(v1 <= v2);
    }
}
""")


class MaxFolding(generic.model.pooling.MaxFolding, gpu.model.layer.Layer):
    block_size = 256

    def __init__(self):
        self._fprop_kernel = _max_folding_module.get_function("fprop_kernel")

    def _fprop(self, X):
        out = pycuda.gpuarray.empty((X.shape[0]//2, X.shape[1]), dtype=np.float32)
        switches = pycuda.gpuarray.empty_like(X)

        rows_per_block = self.__class__.block_size // X.shape[1] + 1
        num_blocks = X.shape[0] // rows_per_block + 1

        self._fprop_kernel(
            X,
            np.int32(X.shape[0]),
            np.int32(X.shape[1]),
            out,
            switches,
            block=(rows_per_block, X.shape[1], 1),
            grid=(num_blocks, 1))

        return out, switches

    # bprop is entirely generic
    # no grads