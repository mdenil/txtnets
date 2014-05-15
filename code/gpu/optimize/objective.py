__author__ = 'mdenil'

import numpy as np

import generic.optimize.objective

import gpu.space

import pycuda.autoinit
import pycuda.gpuarray
import pycuda.compiler
import scikits.cuda.linalg

scikits.cuda.linalg.init()


from generic.optimize.objective import CostMinimizationObjective

###########


class _GaussianEnergy(object):
    def fprop(self, x, y):
        ones = pycuda.gpuarray.zeros((x.shape[1], 1), dtype=x.dtype)
        ones.fill(1.0)
        return 0.5 * scikits.cuda.linalg.dot((x - y)**2, ones)

    def bprop(self, x, y, delta):
        delta_space = gpu.space.GPUSpace.infer(delta, ('b', 'd'))
        delta, delta_space = delta_space.broadcast(delta, d=x.shape[1])

        delta_x = delta * x
        delta_y = delta * y
        return delta_x, delta_y


_contrastive_hinge_loss_module = pycuda.compiler.SourceModule("""
// modifies x in place
__global__ void fprop_kernel(float* x, int N)
{
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < N) {
        const float v = x[thread_id];
        x[thread_id] = v * (float)(v > 0.0f);
    }
}

__global__ void bprop_kernel(
    float* x_clean,
    float* out_clean,
    float* x_noise,
    float* out_noise,
    int N,
    float margin,
    float scale)
{
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < N) {
        float delta = (float)(margin + x_clean[thread_id] - x_noise[thread_id] > 0.0f);
        delta /= scale;
        out_clean[thread_id] = delta * x_clean[thread_id];
        out_noise[thread_id] = delta * -x_noise[thread_id];
    }
}
""")


class _ContrastiveHingeLoss(object):
    block_size = 512

    def __init__(self, margin):
        self.margin = margin
        self.__acquire_device_kernels()

    def fprop(self, x_clean, x_noise):
        out = self.margin + x_clean - x_noise

        elements_per_block = self.__class__.block_size
        num_blocks = out.size // elements_per_block + 1
        block = (elements_per_block, 1, 1)
        grid = (num_blocks, 1)

        self._fprop_kernel(
            out,
            np.int32(out.size),
            block=block,
            grid=grid)

        return out

    def bprop(self, x_clean, x_noise):
        out_clean = pycuda.gpuarray.empty_like(x_clean)
        out_noise = pycuda.gpuarray.empty_like(x_noise)

        elements_per_block = self.__class__.block_size
        num_blocks = x_clean.size // elements_per_block + 1
        block = (elements_per_block, 1, 1)
        grid = (num_blocks, 1)

        self._bprop_kernel(
            x_clean,
            out_clean,
            x_noise,
            out_noise,
            np.int32(x_clean.size),
            np.float32(self.margin),
            np.float32(x_clean.shape[0]),
            block=block,
            grid=grid)

        return out_clean, out_noise

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


class ContrastiveMultilingualEmbeddingObjective(
        generic.optimize.objective.ContrastiveMultilingualEmbeddingObjective):

    Energy = _GaussianEnergy
    LossFunction = _ContrastiveHingeLoss

    def _zeros_like(self, x):
        return pycuda.gpuarray.zeros_like(x)

    def _mean(self, x):
        return pycuda.gpuarray.sum(x) / float(x.size)
