__author__ = 'mdenil'

import numpy as np
import generic.optimize.objective.contrastive_multilingual

import pycuda.autoinit
import pycuda.gpuarray
import pycuda.compiler

import gpu.space
import gpu.allocator

import scikits.cuda.linalg

scikits.cuda.linalg.init()

__all__ = ["ContrastiveMultilingualEmbeddingObjective"]


class GaussianEnergy(object):
    def fprop(self, x, y):
        ones = pycuda.gpuarray.empty((x.shape[1], 1), dtype=x.dtype, allocator=gpu.allocator.global_device_allocator)
        ones.fill(1.0)
        return 0.5 * scikits.cuda.linalg.dot((x - y)**2, ones)

    def bprop(self, x, y, delta):
        delta_space = gpu.space.GPUSpace.infer(delta, ('b', 'd'))
        delta, delta_space = delta_space.broadcast(delta, d=x.shape[1])

        delta_x = delta * (x - y)
        delta_y = delta * (y - x)
        return delta_x, delta_y


_contrastive_hinge_loss_module = pycuda.compiler.SourceModule("""
__global__ void fprop_kernel(float* x_clean, float* x_noise, float margin, int N, float* out)
{
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = thread_id; i < N; i += blockDim.x * gridDim.x) {
        float v = margin + x_clean[i] - x_noise[i];
        out[i] = v * (float)(v > 0.0f) / N;
    }
}

// computes delta_clean.  delta_dirty = -delta_clean
__global__ void bprop_kernel(float* x_clean, float* x_noise, float margin, int N, float delta, float* out)
{
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = thread_id; i < N; i += blockDim.x * gridDim.x) {
        out[i] = delta / N * (float)(margin + x_clean[i] - x_noise[i] > 0.0f);
    }
}
""")


class ContrastiveHingeLoss(object):
    block_size = 512

    def __init__(self, margin):
        self.margin = margin
        self.__acquire_device_kernels()

    def fprop(self, x_clean, x_noise):
        out = pycuda.gpuarray.empty(
            x_clean.shape,
            dtype=x_clean.dtype,
            allocator=gpu.allocator.global_device_allocator)

        elements_per_block = self.__class__.block_size
        num_blocks = out.size // elements_per_block + 1
        block = (elements_per_block, 1, 1)
        grid = (num_blocks, 1)

        self._fprop_kernel(
            x_clean,
            x_noise,
            np.float32(self.margin),
            np.int32(out.size),
            out,
            block=block,
            grid=grid)

        return pycuda.gpuarray.sum(out)

    def bprop(self, x_clean, x_noise, delta):
        out = pycuda.gpuarray.empty(
            x_clean.shape,
            dtype=x_clean.dtype,
            allocator=gpu.allocator.global_device_allocator)

        elements_per_block = self.__class__.block_size
        num_blocks = x_clean.size // elements_per_block + 1
        block = (elements_per_block, 1, 1)
        grid = (num_blocks, 1)

        self._bprop_kernel(
            x_clean,
            x_noise,
            np.float32(self.margin),
            np.int32(x_clean.size),
            np.float32(delta),
            out,
            block=block,
            grid=grid)

        return out, -out

    def __acquire_device_kernels(self):
        self._fprop_kernel = _contrastive_hinge_loss_module.get_function("fprop_kernel")
        self._bprop_kernel = _contrastive_hinge_loss_module.get_function("bprop_kernel")

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['fprop_kernel']
        del state['bprop_kernel']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__acquire_device_kernels()


_square_square_margin_loss_module = pycuda.compiler.SourceModule("""
__global__ void fprop_kernel(float* e_clean, float* e_dirty, float margin, int N, float* out)
{
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = thread_id; i < N; i += blockDim.x * gridDim.x) {
        float pos = e_clean[i];
        float neg = fmaxf(0.0f, margin - e_dirty[i]);

        out[i] = 0.5f / N * (pos * pos + neg * neg);
    }
}

// computes delta_dirty, delta_clean is just delta * e_clean
__global__ void bprop_kernel(float* e_clean, float* e_dirty, float margin, int N, float delta, float* out)
{
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = thread_id; i < N; i += blockDim.x * gridDim.x) {
        out[i] = -delta / N * fmaxf(0.0f, margin - e_dirty[i]);
    }
}
""")


class SquareSquareMarginLoss(object):
    block_size = 512

    def __init__(self, margin):
        self.margin = margin
        self.__acquire_device_kernels()

    def fprop(self, x_clean, x_noise):
        out = pycuda.gpuarray.empty(
            x_clean.shape,
            dtype=x_clean.dtype,
            allocator=gpu.allocator.global_device_allocator)

        elements_per_block = self.__class__.block_size
        num_blocks = out.size // elements_per_block + 1
        block = (elements_per_block, 1, 1)
        grid = (num_blocks, 1)

        self._fprop_kernel(
            x_clean,
            x_noise,
            np.float32(self.margin),
            np.int32(out.size),
            out,
            block=block,
            grid=grid)

        return pycuda.gpuarray.sum(out)

    def bprop(self, x_clean, x_noise, delta):
        out = pycuda.gpuarray.empty(
            x_clean.shape,
            dtype=x_clean.dtype,
            allocator=gpu.allocator.global_device_allocator)

        elements_per_block = self.__class__.block_size
        num_blocks = x_clean.size // elements_per_block + 1
        block = (elements_per_block, 1, 1)
        grid = (num_blocks, 1)

        self._bprop_kernel(
            x_clean,
            x_noise,
            np.float32(self.margin),
            np.int32(x_clean.size),
            np.float32(delta),
            out,
            block=block,
            grid=grid)

        return delta / float(x_clean.shape[0]) * x_clean, out

    def __acquire_device_kernels(self):
        self._fprop_kernel = _square_square_margin_loss_module.get_function("fprop_kernel")
        self._bprop_kernel = _square_square_margin_loss_module.get_function("bprop_kernel")

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['fprop_kernel']
        del state['bprop_kernel']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__acquire_device_kernels()


class ContrastiveMultilingualEmbeddingObjective(
        generic.optimize.objective.contrastive_multilingual.ContrastiveMultilingualEmbeddingObjective):

    Energy = GaussianEnergy
    # LossFunction = ContrastiveHingeLoss
    LossFunction = SquareSquareMarginLoss

    def _zeros_like(self, x):
        return pycuda.gpuarray.zeros(
            x.shape,
            dtype=x.dtype,
            allocator=gpu.allocator.global_device_allocator)
