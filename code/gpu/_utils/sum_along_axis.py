__author__ = 'mdenil'

import numpy as np

import pycuda.autoinit
import pycuda.gpuarray
import pycuda.compiler

__all__ = ['sum_along_axis']

_sum_along_axis_module = pycuda.compiler.SourceModule("""
__global__ void sum_along_axis_kernel(
    float* in,
    int in_rows,
    int in_cols,
    int in_depth,
    int in_row_stride,
    int in_col_stride,
    int in_depth_stride,
    float* out,
    int out_row_stride)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < in_rows && col < in_cols) {
        float total = 0.0f;

        const int in_base = row * in_row_stride + col * in_col_stride;
        for (int i = 0; i < in_depth; ++i) {
            const int in_index = in_base + i * in_depth_stride;
            total += in[in_index];
        }

        const int out_index = row * out_row_stride + col;
        out[out_index] = total;
    }
}
""")

sum_along_axis_kernel = _sum_along_axis_module.get_function("sum_along_axis_kernel")
memory_pool = pycuda.tools.DeviceMemoryPool()


def sum_along_axis(x, axis):
    x_shape = x.shape
    in_rows = prod(x_shape[:axis])
    in_cols = prod(x_shape[axis+1:])
    in_depth = x_shape[axis]
    in_shape = (in_rows, in_depth, in_cols)
    out_shape = (in_rows, in_cols)

    x = x.reshape(in_shape)
    in_strides = tuple(s / x.dtype.itemsize for s in x.strides)

    out = pycuda.gpuarray.empty(shape=out_shape, dtype=x.dtype, allocator=memory_pool.allocate)
    out_strides = tuple(s / out.dtype.itemsize for s in out.strides)

    block_size = 512

    block, grid = block_grid_shape_2d(block_size, in_rows, in_cols)

    sum_along_axis_kernel(
        x,
        np.int32(in_rows),
        np.int32(in_cols),
        np.int32(in_depth),
        np.int32(in_strides[0]),
        np.int32(in_strides[2]),
        np.int32(in_strides[1]),
        out,
        np.int32(out_strides[0]),
        block=block,
        grid=grid)

    out = out.reshape(x_shape[:axis] + x_shape[axis+1:])

    return out


def prod(seq):
    return reduce(lambda x, y: x * y, seq, 1)


def block_grid_shape_2d(block_size, rows, cols):
    if block_size >= cols:
        cols_per_block = cols
        rows_per_block = block_size // cols

        grid_x = rows // rows_per_block + (0 if rows % rows_per_block == 0 else 1)
        grid_y = 1

    else:
        cols_per_block = block_size
        rows_per_block = 1

        grid_x = rows
        grid_y = cols // cols_per_block + (0 if cols % cols_per_block == 0 else 1)

    block = (rows_per_block, cols_per_block, 1)
    grid = (grid_x, grid_y)

    return block, grid
