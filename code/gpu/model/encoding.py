__author__ = 'mdenil'

import numpy as np

import gpu.utils
import gpu.space
import gpu.model.layer

import generic.model.encoding


class DictionaryEncoding(generic.model.encoding.DictionaryEncoding, gpu.model.layer.Layer):
    def _fprop(self, X):
        X = np.vstack([np.atleast_2d(x) for x in X])
        X = gpu.utils.cpu_to_gpu(X.astype(np.int32))
        X_space = gpu.space.GPUSpace.infer(X, ('b', 'w'))

        return X, X_space