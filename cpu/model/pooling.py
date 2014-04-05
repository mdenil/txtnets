__author__ = 'mdenil'

import numpy as np

from cpu import space

class KMaxPooling(object):
    def __init__(self, k):
        self.k = k

    def fprop(self, X, meta):
        working_space = meta['space_below']
        lengths = meta['lengths']

        # FIXME: this is a hack to guarantee statelessness
        X = X.copy()

        d, f, b, w = working_space.get_extents(['d', 'f', 'b', 'w'])

        X, working_space = working_space.transform(X, ['dfb', 'w'])

        padding_mask = lengths.reshape((-1,1)) <= np.arange(working_space.get_extent('w'))
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

        # FIXME: this object should be stateless, but right now I need to know these things for backprop
        # self.space_below = working_space
        # self.k_max_indexes = k_max_indexes
        # self.index_mask = index_mask

        # save these for backprop
        fprop_state = {}
        fprop_state['k_max_indexes'] = k_max_indexes
        fprop_state['index_mask'] = index_mask

        rows = np.vstack([np.arange(working_space.get_extent('dfb'))] * self.k).T

        # print X.shape

        X = X[rows, k_max_indexes]
        X[index_mask] = 0

        working_space = working_space.with_extent(w=self.k)

        # everything has been truncated to length k or smaller
        lengths = np.minimum(lengths, self.k)

        meta['space_above'] = working_space
        meta['lengths'] = lengths

        return X, meta, fprop_state

    def bprop(self, X, delta, meta, fprop_state):

        space_below = meta['space_below']
        space_above = meta['space_above']

        # mask for the values to keep from delta
        index_mask = np.logical_not(fprop_state['index_mask'])
        k_max_indexes = fprop_state['k_max_indexes']

        delta, working_space = space_above.transform(delta, ['dfb', 'w'])

        rows = np.vstack([np.arange(space_below.get_extent('dfb'))] * self.k).T
        back = np.zeros(space_below.shape)
        back, _ = space_below.transform(back, ['dfb', 'w'])

        back[rows[index_mask], k_max_indexes[index_mask]] = delta[index_mask]

        meta['space_below'] = working_space

        return back, meta

    def __repr__(self):
        return "{}(k={})".format(
            self.__class__.__name__,
            self.k)




class SumFolding(object):
    def __init__(self):
        pass

    def fprop(self, X, meta):
        working_space = meta['space_below']

        d, = working_space.get_extents('d')
        assert ( d % 2 == 0 )
        folded_size = d / 2

        X, working_space = working_space.transform(X, ['d', 'b', 'f', 'w'])

        X = X[:folded_size] + X[folded_size:]

        working_space = working_space.with_extent(d=folded_size)
        meta['space_above'] = working_space
        return X, meta

    def bprop(self, X, delta, meta, fprop_state):
        working_space = meta['space_above']

        delta, working_space = working_space.broadcast(delta, d=2)

        meta['space_below'] = working_space
        return delta, meta

    def __repr__(self):
        return "{}()".format(
            self.__class__.__name__)