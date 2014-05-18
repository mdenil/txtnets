__author__ = 'mdenil'


import numpy as np

import pycuda.autoinit
import pycuda.gpuarray
import pycuda.compiler
import pycuda.tools

import reikna.cluda
import reikna.algorithms

import scikits.cuda.linalg
scikits.cuda.linalg.init()

# I don't really understand the implications of making these module level instead of class level.  I think it will
# cause reikna operations to be synchronized between all of the instances using the global thread.
#
# This needs to happen before any pycuda.compiler.SourceModules get run or you will get invalid handle errrors.
cuda_api = reikna.cluda.cuda_api()
cuda_thread = cuda_api.Thread.create()

# Needs to happen after the reikna api and thread creation
import gpu._utils.sum_along_axis


def cpu_to_gpu(X):
    return pycuda.gpuarray.to_gpu(X)


def gpu_to_cpu(X):
    return X.get()


_fliplr_module = pycuda.compiler.SourceModule("""
__global__ void fliplr_kernel(float* in, int n, int m, float* out)
{
    extern __shared__ float buffer[];

    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int c_in = blockIdx.y * blockDim.y + threadIdx.y;
    const int c_out = m - c_in - 1;

    const int r_block = threadIdx.x;
    const int c_block = threadIdx.y;

    if (r < n && c_in < m) {
        buffer[r_block * m + c_block] = in[r * m + c_in];
        __syncthreads();
        out[r * m + c_out] = buffer[r_block * m + c_block];
     }
}
""")


def fliplr(x):
    y = pycuda.gpuarray.empty(x.shape, dtype=x.dtype, allocator=fliplr.memory_pool.allocate)

    block_size = 512
    rows_per_block = block_size // x.shape[1]
    num_blocks = x.shape[0] // rows_per_block + (1 if x.shape[0] % rows_per_block > 0 else 0)

    block = (rows_per_block, x.shape[1], 1)
    grid = (num_blocks, 1)

    fliplr.fliplr_kernel(
        x,
        np.int32(x.shape[0]),
        np.int32(x.shape[1]),
        y,
        shared=int(block_size * x.dtype.itemsize),
        block=block,
        grid=grid)

    return y

fliplr.memory_pool = pycuda.tools.DeviceMemoryPool()
fliplr.fliplr_kernel = _fliplr_module.get_function("fliplr_kernel")


def sum_along_axis(X, space, axis):
    assert axis in space.folded_axes

    # short circut for summing along an axis of size 1
    if space.get_extent(axis) == 1:
        working_space = space.without_axes(axis)
        return working_space.unfold(X), working_space

    target_axis_index = space.folded_axes.index(axis)
    out = gpu._utils.sum_along_axis.sum_along_axis(
        space.fold(X),
        target_axis_index)
    space = space.without_axes(axis)

    return space.unfold(out), space


def transpose(X, axes):
    # The reikna Transpose algorithm doesn't understand the identity transposition
    if list(axes) == range(len(axes)):
        return X

    # Okay, so we're being a bit tricky here.
    #
    # Caching the transpose operations is important because they take a long time to build and compile.  However,
    # to construct a Transpose I need to pass in X, and we need to be careful that the key contains all of the relevant
    # info that could make a transpose different.
    #
    # The Transpose constructor:
    #
    # https://github.com/Manticore/reikna/blob/develop/reikna/algorithms/transpose.py#L89
    #
    # uses axes and X.shape and X.dtype.  It also uses X to construct an Annotation:
    #
    # https://github.com/Manticore/reikna/blob/develop/reikna/core/signature.py#L103
    #
    # which in turn calles Type.from_value:
    #
    # https://github.com/Manticore/reikna/blob/develop/reikna/core/signature.py#L68
    #
    # This factory stores X.dtype, X.shape and X.strides
    key = (X.shape, X.dtype, X.strides, tuple(axes))
    try:
        apply_transpose = transpose.compiled_cache[key]
    except KeyError:
        global cuda_thread
        apply_transpose = reikna.algorithms.Transpose(X, axes).compile(cuda_thread)
        transpose.compiled_cache[key] = apply_transpose

    # TODO: allow out to be passed as a parameter to avoid reallocation
    out = pycuda.gpuarray.empty(apply_transpose.parameter.output.shape, X.dtype)

    apply_transpose(out, X)
    return out

transpose.compiled_cache = {}


_broadcast_module = pycuda.compiler.SourceModule("""
__global__ void broadcast_kernel(
    float* source, int *source_shape, int* source_stride, int rank, float *dest, int *dest_shape, int *dest_stride,
    int dest_size)
{
    // grid strided kernel
    for (int dest_index = blockIdx.x * blockDim.x + threadIdx.x;
        dest_index < dest_size;
        dest_index += blockDim.x * gridDim.x) {

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
_broadcast_kernel = _broadcast_module.get_function("broadcast_kernel")

def broadcast(x, expanded_shape, out=None, block_size=512):
    assert len(x.shape) == len(expanded_shape)

    rank = len(expanded_shape)

    try:
        x_shape, x_stride, y_shape, y_stride = broadcast._shape_cache[rank]
    except KeyError:
        x_shape = pycuda.gpuarray.empty(rank, dtype=np.int32)
        x_stride = pycuda.gpuarray.empty(rank, dtype=np.int32)
        y_shape = pycuda.gpuarray.empty(rank, dtype=np.int32)
        y_stride = pycuda.gpuarray.empty(rank, dtype=np.int32)
        broadcast._shape_cache[rank] = x_shape, x_stride, y_shape, y_stride

    x_shape.set(np.asarray(x.shape, dtype=np.int32))
    x_stride.set(np.asarray(x.strides, dtype=np.int32))

    if out is None:
        y = pycuda.gpuarray.empty(expanded_shape, dtype=np.float32)
    else:
        y = out

    y_shape.set(np.asarray(y.shape, dtype=np.int32))
    y_stride.set(np.asarray(y.strides, dtype=np.int32))

    global _broadcast_kernel

    max_grid_size = 65535

    _broadcast_kernel(
        x,
        x_shape,
        x_stride,
        np.int32(len(x.shape)),
        y,
        y_shape,
        y_stride,
        np.int32(y.size),
        block=(block_size, 1, 1),
        grid=(min(y.size // block_size + 1, max_grid_size), 1))

    return y


broadcast._shape_cache = {}


# def stack(X, axis, times):
#     """
#     Stack times copies of X along axis.
#
#     Right now this is implemented by permuting the target axis to the left, stacking vertically, and then permuting
#     back, so it will be significantly more efficient to stack along the first axis than a different one.
#     """
#     # store the shape so we can restore it later
#     original_shape = X.shape
#
#     # transpose the axis to be first
#     perm = [axis] + [ax for ax in range(len(X.shape)) if ax != axis]
#     assert len(perm) == len(X.shape)
#     X = transpose(X, perm)
#
#     target_permuted_shape = (X.shape[0] * times, ) + X.shape[1:]
#
#     # The 2d shape of X.  From now on we assume X is in row-major order.
#     flat_shape = (X.shape[0], int(np.prod(X.shape[1:])))
#     X = X.reshape(flat_shape)
#
#     # stack times copies of X vertically
#     Y = pycuda.gpuarray.empty((flat_shape[0]*times, flat_shape[1]), dtype=X.dtype)
#     # TODO: use Y to Y copies to reduce the number of copies
#     for t in xrange(times):
#         _copy_paste_2d(X, 0, 0, flat_shape[0], flat_shape[1], Y, t * flat_shape[0], 0)
#
#     # restore the extra dimensions in the stacked data
#     Y = Y.reshape(target_permuted_shape)
#
#     # invert the permutation so Y has the same axis order as the original X
#     inverse_perm = tuple(perm.index(i) for i in xrange(len(perm)))
#     Y = transpose(Y, inverse_perm)
#
#     return Y


_copy_2d = pycuda.driver.Memcpy2D()

def stack(X, axis, times):
    """
    Stack times copies of X along axis.

    Right now this is implemented by permuting the target axis to the left, stacking vertically, and then permuting
    back, so it will be significantly more efficient to stack along the first axis than a different one.
    """

    # transpose the axis to be first
    perm = [axis] + [ax for ax in range(len(X.shape)) if ax != axis]
    assert len(perm) == len(X.shape)
    X = transpose(X, perm)

    permuted_shape = X.shape
    target_permuted_shape = (X.shape[0] * times, ) + X.shape[1:]

    # flatten X to 2d
    flat_shape = (X.shape[0], int(np.prod(X.shape[1:])))
    X = X.reshape(flat_shape)

    # allocate flat memory for the broadcasted result
    Y = pycuda.gpuarray.empty((flat_shape[0]*times, flat_shape[1]), dtype=X.dtype)


    sh, sw = X.shape
    dh, dw = Y.shape
    sr = 0
    sc = 0
    nr = flat_shape[0]
    nc = flat_shape[1]
    dr = 0
    dc = 0

    global _copy_2d

    # Set and describe copy source
    _copy_2d.set_src_device(X.gpudata)
    _copy_2d.src_x_in_bytes = sc * X.dtype.itemsize
    _copy_2d.src_y = sr
    _copy_2d.src_pitch = sw * X.dtype.itemsize

    # Set and describe copy destination
    _copy_2d.set_dst_device(Y.gpudata)
    _copy_2d.dst_x_in_bytes = dc * Y.dtype.itemsize
    _copy_2d.dst_y = dr
    _copy_2d.dst_pitch = dw * Y.dtype.itemsize

    # Describe copy extent
    _copy_2d.width_in_bytes = nc * Y.dtype.itemsize
    _copy_2d.height = nr

    # move the first copy of X
    _copy_2d(aligned=True)

    # move the source pointer to start copying Y to itself
    _copy_2d.set_src_device(Y.gpudata)

    dr += sh
    while dr < dh:
        _copy_2d.dst_y = dr
        _copy_2d.height = min(nr, dh - dr)

        _copy_2d(aligned=True)

        dr += _copy_2d.height
        nr += _copy_2d.height

    # restore the extra dimensions in the stacked data
    Y = Y.reshape(target_permuted_shape)

    # invert the permutation so Y has the same axis order as the original X
    inverse_perm = tuple(perm.index(i) for i in xrange(len(perm)))
    Y = transpose(Y, inverse_perm)

    return Y



def _copy_paste_2d(src, sr, sc, nr, nc, dst, dr, dc):
    # Copies a rectangular region of entries from src into dst.
    #
    # The rectangle to copy has upper right corner (sr, sc) and lower right corner (sr + nr, sc + nc).  The rectangle
    # is treated as a half open interval in both dimensions, so
    #
    #   _copy_paste_2d(src, 0, 0, src.shape[0], src.shape[1], dst, 0, 0)
    #
    # is the correct way to copy all of the values in src to dst.
    #
    # The target of the copy is a 2d region in dst, with upper left corner (dr, dc).
    #
    # This function assumes src and dst are stored in row major order.

    assert src.dtype.itemsize == dst.dtype.itemsize

    sh, sw = src.shape
    dh, dw = dst.shape

    # http://documen.tician.de/pycuda/driver.html?highlight=memcpy2d#pycuda.driver.Memcpy2D
    copy = pycuda.driver.Memcpy2D()

    # Set and describe copy source
    copy.set_src_device(src.gpudata)
    copy.src_x_in_bytes = sc * src.dtype.itemsize
    copy.src_y = sr
    copy.src_pitch = sw * src.dtype.itemsize

    # Set and describe copy destination
    copy.set_dst_device(dst.gpudata)
    copy.dst_x_in_bytes = dc * dst.dtype.itemsize
    copy.dst_y = dr
    copy.dst_pitch = dw * dst.dtype.itemsize

    # Describe copy extent
    copy.width_in_bytes = nc * src.dtype.itemsize
    copy.height = nr

    copy(aligned=True)
