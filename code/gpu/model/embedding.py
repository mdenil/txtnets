__author__ = 'mdenil'

import numpy as np

import gpu.utils
import gpu.model.layer
import generic.model.embedding

import pycuda.autoinit
import pycuda.compiler

_embedding_module = pycuda.compiler.SourceModule("""
__global__ void fprop_kernel(float* E, int e_n, int e_m, int* X, int x_n, float* out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < x_n && j < e_m) {
        out[i * e_m + j] = E[X[i] * e_m + j];
    }
}

// one thread per row
__global__ void bprop_kernel(float* delta, float* Y, int N, int M, float* out)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N) {
        out[row] = 0.0f;

        for (int j = 0; j < M; ++j) {
            out[row] += delta[row * M + j] * Y[row * M + j];
        }
    }
}

__global__ void grads_kernel(float* delta, int N, int M, int* X, float* out, int padding_row)
{
    // out needs to be zeroed before this kernel is called

    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < M) {
        if (X[row] != padding_row) {
            // use an atomic op because multiple Xs can index the same row of out
            atomicAdd(out + X[row] * M + col, delta[row * M + col]);
        }
    }
}
""")


class WordEmbedding(generic.model.embedding.WordEmbedding, gpu.model.layer.Layer):
    block_size = 512

    def __init__(self, *args, **kwargs):
        super(WordEmbedding, self).__init__(*args, **kwargs)
        self.E = gpu.utils.cpu_to_gpu(self.E.astype(np.float32))
        self.__acquire_device_kernels()

    def _fprop(self, X):
        out = pycuda.gpuarray.empty((X.size, self.E.shape[1]), dtype=np.float32)

        rows_per_block = self.__class__.block_size // self.E.shape[1]
        num_blocks = X.shape[0] // rows_per_block + 1

        self._fprop_kernel(
            self.E,
            np.int32(self.E.shape[0]),
            np.int32(self.E.shape[1]),
            X,
            np.int32(X.size),
            out,
            block=(rows_per_block, self.E.shape[1], 1),
            grid=(num_blocks, 1))

        return out

    def _bprop(self, delta, Y):
        out = pycuda.gpuarray.empty((Y.shape[0],), dtype=np.float32)

        # this kernel uses one thread per row because it needs to accumulate
        rows_per_block = self.__class__.block_size
        num_blocks = delta.shape[0] // rows_per_block + 1

        assert delta.shape[0] == Y.shape[0]

        self._bprop_kernel(
            delta,
            Y,
            np.int32(delta.shape[0]),
            np.int32(delta.shape[1]),
            out,
            block=(rows_per_block, 1, 1),
            grid=(num_blocks, 1))

        return out

    def _grads(self, delta, X):
        grad_E = pycuda.gpuarray.zeros_like(self.E)

        rows_per_block = self.__class__.block_size // self.E.shape[1]
        num_blocks = self.E.shape[0] // rows_per_block + 1

        self._grads_kernel(
            delta,
            np.int32(delta.shape[0]),
            np.int32(delta.shape[1]),
            X,
            grad_E,
            np.int32(self.padding),
            block=(rows_per_block, self.E.shape[1], 1),
            grid=(num_blocks, 1))

        return [grad_E]

    def __acquire_device_kernels(self):
        self._fprop_kernel = _embedding_module.get_function("fprop_kernel")
        self._bprop_kernel = _embedding_module.get_function("bprop_kernel")
        self._grads_kernel = _embedding_module.get_function("grads_kernel")

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_fprop_kernel']
        del state['_bprop_kernel']
        del state['_grads_kernel']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__acquire_device_kernels()

    def move_to_cpu(self):
        from gpu.model.host_device_component_mapping import get_cpu_analog
        cpu_class = get_cpu_analog(self.__class__)

        return cpu_class(
            dimension=self.dimension,
            vocabulary_size=self.vocabulary_size,
            padding=self.padding,
            E=gpu.utils.gpu_to_cpu(self.E))