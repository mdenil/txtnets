__author__ = 'mdenil'

import numpy as np
from cpu import space


# FIXME: Make this actually generic, right now it implements the CPU version.  It lives here instead
# of in the cpu package because the GPU version of KMaxPooling right now just transfers data to the CPU
# and calls this
class KMaxPooling(object):
    def __init__(self, k, k_dynamic=None):
        self.k = k
        self.k_dynamic = k_dynamic

    def fprop(self, X, meta):
        working_space = meta['space_below']
        lengths = meta['lengths']

        # FIXME: this is a hack to guarantee statelessness
        X = X.copy()

        b, d, f, w = working_space.get_extents(['b', 'd', 'f', 'w'])

        working_axes = (('b', 'd', 'f'), 'w')

        X, working_space = working_space.transform(X, working_axes)
        # X, working_space = working_space.transform(X, [('d','f','b'), 'w'])

        fprop_state = {
            "space_below": working_space,
            "lengths_below": lengths,
        }

        padding_mask = lengths.reshape((-1,1)) <= np.arange(working_space.get_extent('w'))
        padding_space = space.CPUSpace.infer(padding_mask, ['b', 'w'])
        padding_mask, padding_space = padding_space.transform(padding_mask, working_axes, d=d, f=f)

        if not self.k_dynamic:
            # static pooling
            index_mask = lengths.reshape((-1,1)) <= np.arange(self.k)[::-1]
            index_space = space.CPUSpace.infer(index_mask, ['b', 'w'])
            index_mask, index_space = index_space.transform(index_mask, working_axes, d=d, f=f)
            k = self.k
            ks = self.k

        else:
            # dynamic pooling
            ks = np.ceil(lengths * self.k_dynamic)
            ks[ks < self.k] = np.minimum(lengths[ks < self.k], self.k)
            max_k = np.max(ks)
            max_k_rank = np.vstack([np.arange(max_k)[::-1]]*len(lengths))
            index_mask = ks.reshape((-1,1)) <= max_k_rank
            k = int(max_k)

        X[padding_mask] = -np.inf

        k_max_indexes = np.argsort(X, axis=1)
        k_max_indexes = k_max_indexes[:, -k:]
        k_max_indexes[index_mask] = np.iinfo(k_max_indexes.dtype).max
        k_max_indexes.sort(axis=1)
        index_mask = (k_max_indexes == np.iinfo(k_max_indexes.dtype).max)
        k_max_indexes[index_mask] = 0

        # save these for backprop
        fprop_state['k_max_indexes'] = k_max_indexes
        fprop_state['index_mask'] = index_mask
        fprop_state['k'] = k

        rows = np.vstack([np.arange(working_space.get_extent(('b', 'd', 'f')))] * k).T

        X = X[rows, k_max_indexes]
        X[index_mask] = 0

        working_space = working_space.with_extents(w=k)

        # everything has been truncated to length k or smaller
        lengths = np.minimum(lengths, ks)

        meta['space_above'] = working_space
        meta['lengths'] = lengths

        return X, meta, fprop_state

    def bprop(self, delta, meta, fprop_state):

        space_below = fprop_state['space_below']
        space_above = meta['space_above']

        # mask for the values to keep from delta
        index_mask = np.logical_not(fprop_state['index_mask'])
        k_max_indexes = fprop_state['k_max_indexes']

        working_axes = (('b', 'd', 'f'), 'w')

        delta, working_space = space_above.transform(delta, working_axes)

        rows = np.vstack([np.arange(space_below.get_extent(('b', 'd', 'f')))] * fprop_state['k']).T
        back = np.zeros(space_below.shape)
        back, _ = space_below.transform(back, working_axes)

        back[rows[index_mask], k_max_indexes[index_mask]] = delta[index_mask]

        meta['space_below'] = working_space.with_extents(w=space_below.get_extent('w'))
        meta['lengths'] = fprop_state['lengths_below']

        return back, meta

    def __repr__(self):
        return "{}(k={}, k_dynamic={})".format(
            self.__class__.__name__,
            self.k,
            self.k_dynamic)


# TODO: Combine shared code for SumFolding and MaxFolding
class SumFolding(object):
    def fprop(self, X, meta):
        working_space = meta['space_below']

        d = working_space.get_extent('d')
        assert d % 2 == 0

        X, working_space = working_space.transform(X, ('b', 'd2', 'd', 'f', 'w'))
        working_space = working_space.with_extents(d=d//2, d2=2)
        X = working_space.fold(X)

        X, working_space = self._fprop(X, working_space)

        meta['space_above'] = working_space
        return X, meta, {}

    def bprop(self, delta, meta, fprop_state):
        working_space = meta['space_above']

        delta, working_space = working_space.broadcast(delta, d=2)

        meta['space_below'] = working_space
        return delta, meta

    def __repr__(self):
        return "{}()".format(
            self.__class__.__name__)


class MaxFolding(object):
    def fprop(self, X, meta):
        working_space = meta['space_below']

        d = working_space.get_extent('d')
        assert d % 2 == 0

        X, working_space = working_space.transform(X, ('d', ('b', 'f', 'w')))

        Y, switches = self._fprop(X)
        switches_space = working_space
        working_space = working_space.with_extents(d=Y.shape[working_space.axes.index('d')])

        fprop_state = {
            'switches': switches,
            'switches_space': switches_space,
        }

        meta['space_above'] = working_space
        return Y, meta, fprop_state

    def bprop(self, delta, meta, fprop_state):
        working_space = meta['space_above']
        switches = fprop_state['switches']
        switches_space = fprop_state['switches_space']

        delta, working_space = working_space.broadcast(delta, d=2)
        switches, switches_space = switches_space.transform(switches, working_space.axes)

        delta *= switches

        meta['space_below'] = working_space
        return delta, meta

    def __repr__(self):
        return "{}()".format(
            self.__class__.__name__)