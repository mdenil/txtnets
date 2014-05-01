__author__ = 'mdenil'

import numpy as np

import gpu.utils
import gpu.space

import generic.optimize.data_provider


class LabelledSequenceMinibatchProvider(
    generic.optimize.data_provider.LabelledSequenceMinibatchProvider):
    """
    This class doesn't actually provide GPU Xs.  It provides lists of X's on the CPU.
    It DOES describe the Xs with a GPU space though, which might be confusing.

    You're supposed to send the output of this provider into an encoder which will actually
    construct GPU matrices for your Xs.

    This class does provide Ys that live on the GPU.
    """

    def next_batch(self):
        X_batch, Y_batch, meta = super(LabelledSequenceMinibatchProvider, self).next_batch()

        meta['space_below'] = gpu.space.GPUSpace(
            meta['space_below'].axes,
            meta['space_below'].extents)

        Y_batch = gpu.utils.cpu_to_gpu(Y_batch.astype(np.int32))

        return X_batch, Y_batch, meta