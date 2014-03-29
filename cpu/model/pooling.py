__author__ = 'mdenil'

import numpy as np

class KMaxPooling(object):
    def __init__(self, k):
        self.k = k
        self.input_axes = ['d', 'f', 'b', 'w']
        self.output_axes = ['d', 'f', 'b', 'w']

    def fprop(self, X, **meta):
        d, f, b, w = X.shape

        X = np.reshape(
            X,
            (d * f * b, w)
        )

        # padding_mask has axes [b, w]
        padding_mask = meta['lengths'].reshape((-1,1)) <= np.arange(w)
        # stack to [d*f*b, w] to match X
        padding_mask = np.vstack([padding_mask] * (d * f))

        X[padding_mask] = -np.inf

        k_max_indexes = np.argsort(X, axis=1)
        k_max_indexes = k_max_indexes[:,-self.k:]
        k_max_indexes.sort(axis=1)

        rows = np.vstack([np.arange(d * f * b)] * self.k).T

        X = X[rows, k_max_indexes]

        X = np.reshape(
            X,
            (d, f, b, self.k)
        )

        return X, meta

    def __repr__(self):
        return "{}(k={})".format(
            self.__class__.__name__,
            self.k)




class SumFolding(object):
    def __init__(self):
        self.input_axes = ['d', 'b', 'f', 'w']
        self.output_axes = ['d', 'b', 'f', 'w']

    def fprop(self, X, **meta):
        d, b, f, w = X.shape

        assert ( d % 2 == 0 )
        folded_size = d / 2

        X = X[:folded_size] + X[folded_size:]

        X = np.reshape(
            X, (folded_size, b, f, w)
        )

        return X, meta

    def __repr__(self):
        return "{}()".format(
            self.__class__.__name__)