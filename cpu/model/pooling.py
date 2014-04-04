__author__ = 'mdenil'

import numpy as np

from cpu import space

class KMaxPooling(object):
    def __init__(self, k):
        self.k = k

    def fprop(self, X, **meta):
        # d, f, b, w = X.shape
        data_space = meta['data_space']
        lengths = meta['lengths']

        d, f, b, w = data_space.get_extents(['d', 'f', 'b', 'w'])

        X, data_space = data_space.transform(X, ['dfb', 'w'])

        padding_mask = lengths.reshape((-1,1)) <= np.arange(data_space.get_extent('w'))
        padding_space = space.Space.infer(padding_mask, ['b', 'w'])
        padding_mask, padding_space = padding_space.transform(padding_mask, ['dfb', 'w'], d=d, f=f)

        index_mask = lengths.reshape((-1,1)) <= np.arange(self.k)[::-1]
        index_space = space.Space.infer(index_mask, ['b', 'w'])
        index_mask, index_space = index_space.transform(index_mask, ['dfb', 'w'], d=d, f=f)

        X[padding_mask] = -np.inf

        k_max_indexes = np.argsort(X, axis=1)
        k_max_indexes = k_max_indexes[:,-self.k:]
        k_max_indexes[index_mask] = np.iinfo(k_max_indexes.dtype).max
        k_max_indexes.sort(axis=1)
        index_mask = (k_max_indexes == np.iinfo(k_max_indexes.dtype).max)
        k_max_indexes[index_mask] = 0

        rows = np.vstack([np.arange(data_space.get_extent('dfb'))] * self.k).T

        X = X[rows, k_max_indexes]
        X[index_mask] = 0

        data_space = data_space.set_extent(w=self.k)

        # everything has been truncated to length k or smaller
        lengths = np.minimum(lengths, self.k)

        meta['data_space'] = data_space
        meta['lengths'] = lengths

        return X, meta

    def __repr__(self):
        return "{}(k={})".format(
            self.__class__.__name__,
            self.k)




class SumFolding(object):
    def __init__(self):
        pass

    def fprop(self, X, **meta):
        data_space = meta['data_space']

        d, = data_space.get_extents('d')
        assert ( d % 2 == 0 )
        folded_size = d / 2

        X, data_space = data_space.transform(X, ['d', 'b', 'f', 'w'])

        X = X[:folded_size] + X[folded_size:]

        data_space = data_space.set_extent(d=folded_size)
        meta['data_space'] = data_space
        return X, meta

    def __repr__(self):
        return "{}()".format(
            self.__class__.__name__)