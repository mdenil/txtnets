__author__ = 'mdenil'

import numpy as np

import pycuda
import pycuda.autoinit
import pycuda.gpuarray
import pycuda.compiler


broadcast_module = pycuda.compiler.SourceModule("""
__global__ void fliplr_kernel(
    float* source, int *source_shape, int* source_stride, int rank, float *dest, int *dest_shape, int *dest_stride,
    int dest_size)
{
    const int dest_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (dest_index < dest_size) {

        // figure out where the source element lives
        int source_index = 0;
        for (int i = 0; i < rank; ++i) {
            int dest_coordinate = dest_index / (dest_stride[i] / sizeof(float)) % dest_shape[i];
            int source_coordinate = dest_coordinate % source_shape[i];
            source_index += source_coordinate * source_stride[i] / sizeof(float);
        }

        float* dest_element = dest + dest_index;
        float* source_element = source + source_index;

        *dest_element = *source_element;
    }
}
""")

broadcast_kernel = broadcast_module.get_function("broadcast_kernel")


def broadcast(x, expanded_shape, out=None, block_size=256):
    assert len(x.shape) == len(expanded_shape)

    x_shape = pycuda.gpuarray.to_gpu(np.asarray(x.shape, dtype=np.int32))
    x_stride = pycuda.gpuarray.to_gpu(np.asarray(x.strides, dtype=np.int32))

    if not out:
        y = pycuda.gpuarray.empty(expanded_shape, dtype=np.float32)
    else:
        y = out
    y_shape = pycuda.gpuarray.to_gpu(np.asarray(y.shape, dtype=np.int32))
    y_stride = pycuda.gpuarray.to_gpu(np.asarray(y.strides, dtype=np.int32))

    global broadcast_kernel

    broadcast_kernel(
        x.gpudata,
        x_shape.gpudata,
        x_stride.gpudata,
        np.int32(len(x.shape)),
        y.gpudata,
        y_shape.gpudata,
        y_stride.gpudata,
        np.int32(y.size),
        block=(block_size, 1, 1),
        grid=(y.size / block_size + 1, 1))

    return y


###########

import unittest

def broadcast_kernel_python(source, source_shape, source_stride, rank, dest, dest_shape, dest_stride, dest_size):
    for dest_index in xrange(dest_size):
        dest_coordinate = dest_index / (dest_stride / dest.itemsize) % dest_shape
        source_coordinate = dest_coordinate % source_shape
        source_index = np.sum(source_coordinate * (source_stride / source.itemsize))
        dest[dest_index] = source[source_index]


def broadcast_python(x, expanded_shape):

    y = np.empty(expanded_shape, dtype=x.dtype)

    broadcast_kernel_python(
        x.ravel(),
        np.asarray(x.shape),
        np.asarray(x.strides),
        len(x.shape),
        y.ravel(),
        np.asarray(y.shape),
        np.asarray(y.strides),
        y.size)

    return y


class TestBroadcast(unittest.TestCase):
    def setUp(self):
        pass

    def test_broadcast_kernel(self):
        a_cpu = np.arange(27).reshape((3, 3, 3)).astype(np.float32)
        a_gpu = pycuda.gpuarray.to_gpu(a_cpu)

        b_cpu = broadcast_python(a_cpu, (6, 6, 3))
        b_gpu = broadcast(a_gpu, (6, 6, 3))

        self.assertTrue(np.allclose(b_cpu, b_gpu.get()))