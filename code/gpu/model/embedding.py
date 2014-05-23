__author__ = 'mdenil'

import numpy as np

import gpu.utils
import gpu.model.layer
import generic.model.embedding

import pycuda.autoinit
import pycuda.compiler

import gpu.allocator


_embedding_module = pycuda.compiler.SourceModule("""
__global__ void fprop_kernel(float* E, int* X, int B, int D, int W, int V, float* out)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    const int w = blockIdx.y * blockDim.y + threadIdx.y;

    if (b < B && w < W) {
        for (int d = 0; d < D; ++d) {
            out[b*D*W + d*W + w] = E[d*V + X[b*W + w]];
        }
    }
}

__global__ void grads_kernel(float* delta, int* X, int B, int D, int W, int V, float* out, int v_padding)
{
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    const int b = thread_id / (D * W);
    const int d = (thread_id % (D * W)) / W;
    const int w = thread_id % W;

    if (b < B && d < D && w < W) {
        int v = X[b*W + w];
        if (v != v_padding) {
            atomicAdd(out + d*V + v, delta[b*D*W + d*W + w]);
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
        # X is formatted as b, d=1, w
        # out is b, d, w
        # E is d, vocab_size

        b, _, w = X.shape
        d, v = self.E.shape

        out = pycuda.gpuarray.empty((b, d, w), dtype=np.float32, allocator=gpu.allocator.global_device_allocator)

        # kernel blocks are formatted to have one thread per element of X
        # each thread loops over d.

        if self.__class__.block_size >= w:
            sentences_per_block = self.__class__.block_size // w
            num_blocks = b // sentences_per_block + 1

            block = (sentences_per_block, w, 1)
            grid = (num_blocks, 1)
        else:
            blocks_per_sentence = w // self.__class__.block_size + 1
            num_blocks = (b * blocks_per_sentence, 1)

            block = (1, blocks_per_sentence, 1)
            grid = num_blocks

        self._fprop_kernel(
            self.E,
            X,
            np.int32(b),
            np.int32(d),
            np.int32(w),
            np.int32(v),
            out,
            block=block,
            grid=grid)

        return out

    def _bprop(self, delta, Y, space):
        delta *= Y
        delta, _ = gpu.utils.sum_along_axis(delta, space, 'd')
        return delta

    def _grads(self, delta, X):
        grad_E = pycuda.gpuarray.zeros(self.E.shape, dtype=np.float32, allocator=gpu.allocator.global_device_allocator)

        b, _, w = X.shape
        d, v = self.E.shape

        block = (self.__class__.block_size, 1, 1)
        grid = (delta.size // self.__class__.block_size + 1, 1)

        self._grads_kernel(
            delta,
            X,
            np.int32(b),
            np.int32(d),
            np.int32(w),
            np.int32(v),
            grad_E,
            np.int32(self.padding),
            block=block,
            grid=grid)

        return [grad_E]

    def __acquire_device_kernels(self):
        self._fprop_kernel = _embedding_module.get_function("fprop_kernel")
        self._grads_kernel = _embedding_module.get_function("grads_kernel")

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_fprop_kernel']
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