__author__ = 'mdenil'


import numpy as np

import pycuda.autoinit
import pycuda.gpuarray

import reikna.cluda
import reikna.algorithms

def cpu_to_gpu(X):
    return pycuda.gpuarray.to_gpu(X)

def gpu_to_cpu(X):
    return X.get()


# I don't really understand the implications of making these module level instead of class level.  I think it will
# cause reikna operations to be synchronized between all of the instances using the global thread.
cuda_api = reikna.cluda.cuda_api()
cuda_thread = cuda_api.Thread.create()


def transpose(X, axes):
    # The reikna Transpose algorithm doesn't understand the identity transposition
    if list(axes) == range(len(axes)):
        return X

    # TODO: can this be cached? does it matter?
    global cuda_thread
    apply_transpose = reikna.algorithms.Transpose(X, axes).compile(cuda_thread)

    # TODO: allow out to be passed as a parameter to avoid reallocation
    out = pycuda.gpuarray.empty(apply_transpose.parameter.output.shape, X.dtype)

    apply_transpose(out, X)
    return out


def stack(X, axis, times):
    """
    Stack times copies of X along axis.

    Right now this is implemented by permuting the target axis to the left, stacking vertically, and then permuting
    back, so it will be significantly more efficient to stack along the first axis than a different one.
    """
    # store the shape so we can restore it later
    original_shape = X.shape

    # transpose the axis to be first
    perm = [axis] + [ax for ax in range(len(X.shape)) if ax != axis]
    assert len(perm) == len(X.shape)
    X = transpose(X, perm)

    target_permuted_shape = (X.shape[0] * times, ) + X.shape[1:]

    # The 2d shape of X.  From now on we assume X is in row-major order.
    flat_shape = (X.shape[0], int(np.prod(X.shape[1:])))
    X = X.reshape(flat_shape)

    # stack times copies of X vertically
    Y = pycuda.gpuarray.empty((flat_shape[0]*times, flat_shape[1]), dtype=X.dtype)
    # TODO: use Y to Y copies to reduce the number of copies
    for t in xrange(times):
        _copy_paste_2d(X, 0, 0, flat_shape[0], flat_shape[1], Y, t * flat_shape[0], 0)

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
