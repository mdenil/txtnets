__author__ = 'mdenil'

import pycuda.autoinit
import pycuda
import pycuda.gpuarray

import scikits.cuda.fft

# for dtypes
import numpy as np


class FFTConv1D(object):
    def __init__(self, use_buffer_cache=True):
        self.use_buffer_cache = use_buffer_cache

        self._buffer_cache = {}
        self._plan_cache = {}

    def conv(self, X, K, mode='full'):
        # always convolves along axis=1

        assert mode in ['full', 'valid']

        xw = X.shape[1]
        kw = K.shape[1]

        if mode == 'valid':
            assert xw >= kw

        if X.shape[0] != K.shape[0]:
            raise ValueError("Batch size must match for convolution. (X.shape={}, K.shape={})".format(
                X.shape, K.shape))

        # pad

        if xw >= kw:
            padded_shape = (X.shape[0], _power_2_upper_bound(xw + kw - 1))
        else:
            padded_shape = (X.shape[0], _power_2_upper_bound((kw - xw) + kw - 1))

        X_out = self._get_cached_buffer('X_out', padded_shape, np.float32)
        K_out = self._get_cached_buffer('K_out', padded_shape, np.float32)

        X = _pad_right(X, padded_shape[1], out=X_out)
        K = _pad_right(K, padded_shape[1], out=K_out)

        # plan

        forward_plan = self._get_cached_plan(X.shape, np.float32, np.complex64)
        backward_plan = self._get_cached_plan(X.shape, np.complex64, np.float32)

        # compute

        # The cuda fft only computes non-redundant coefficients for real -> complex ffts
        fft_shape = (X.shape[0], X.shape[1] / 2 + 1)
        X_fft = self._get_cached_buffer('X_fft', fft_shape, np.complex64)
        K_fft = self._get_cached_buffer('K_fft', fft_shape, np.complex64)

        scikits.cuda.fft.fft(X, X_fft, forward_plan)
        scikits.cuda.fft.fft(K, K_fft, forward_plan)
        X_fft *= K_fft

        # This is really weird, but scaling inside the ifft is a lot slower than scaling manually.
        # I'm not the only one who has encountered this:
        # https://groups.google.com/forum/#!topic/theano-users/6xiFFpBBDq0
        scikits.cuda.fft.ifft(X_fft, X, backward_plan, False)
        X /= X.size/backward_plan.batch

        # trim

        if mode == 'full':
            Y = _extract_columns(X, 0, xw + kw - 1)
        elif mode == 'valid':
            Y = _extract_columns(X, kw - 1, xw)

        return Y

    def clear_cache(self):
        """
        Empties the internal caches.  If caching is turned off this is a no-op.
        """

        if not self.use_cache:
            return

        self._plan_cache = {}

        # be really explicit about freeing gpu data
        for buf in self._buffer_cache.values():
            buf.gupdata.free()
        self._buffer_cache = {}

    def _get_cached_plan(self, shape, from_dtype, to_dtype):
        """
        shape = (batch_size, length_of_each_transform).  This is the .shape of the matrix that will be transformed.
        from_dtype and to_dtype are the source and target types for the FFT
        """
        key = (shape, from_dtype, to_dtype)

        try:
            plan = self._plan_cache[key]
        except KeyError:
            plan = scikits.cuda.fft.Plan((shape[1],), from_dtype, to_dtype, shape[0])
            self._plan_cache[key] = plan

        return plan

    def _get_cached_buffer(self, name, shape, dtype):
        """
        name = a name for this buffer
        shape = shape of the buffer to allocate
        dtype = type of the buffer to allocate

        Newly created buffers are always zeroed, but cached buffers are untouched (zero them yourself if you need that).

        Buffers are uniquely identified by the name, shape and dtype.  Using different names will let you have more than
        one buffer with the same name and dtype.
        """
        key = (name, shape, dtype)

        try:
            buf = self._buffer_cache[key]
        except KeyError:
            buf = pycuda.gpuarray.zeros(shape, dtype=dtype)
            if self.use_buffer_cache:
                self._buffer_cache[key] = buf

        return buf


def _pad_right(X, width, out=None):
    b, w = X.shape

    if not out:
        out = pycuda.gpuarray.zeros((b, width), dtype=X.dtype)
    else:
        assert out.shape[1] == width
        out.fill(0.0)

    copy = pycuda.driver.Memcpy2D()
    copy.set_src_device(X.gpudata)
    copy.src_pitch = w * X.dtype.itemsize  # row width at source
    copy.width_in_bytes = w * X.dtype.itemsize  # width to copy

    copy.set_dst_device(out.gpudata)
    copy.dst_pitch = width * X.dtype.itemsize  # row width at dest
    copy.height = b  # number of rows to copy

    copy(aligned=True)

    return out


def _extract_columns(X, start, end):
    # X[:, start:end], but guarantees the output will be contiguous

    b, w = X.shape

    # I've taken this out for now because I'm not 100% sure it won't result in trying to use the same buffer for two
    # things at once somewhere down the line.
    # Avoid the copy if we're really taking all the columns
    # if w == end - start:
    #     return X

    Y = pycuda.gpuarray.empty((b, end - start), dtype=X.dtype)

    copy = pycuda.driver.Memcpy2D()
    copy.set_src_device(X.gpudata)
    copy.src_x_in_bytes = start * X.dtype.itemsize
    copy.src_pitch = w * X.dtype.itemsize
    copy.width_in_bytes = (end - start) * X.dtype.itemsize

    copy.set_dst_device(Y.gpudata)
    copy.dst_pitch = (end - start) * X.dtype.itemsize
    copy.height = b

    copy(aligned=True)

    return Y


def _power_2_upper_bound(x):
    return 1 << (x - 1).bit_length()