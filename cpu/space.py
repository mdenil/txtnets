__author__ = 'mdenil'

import numpy as np
from collections import OrderedDict

def _fold_axes(unfolded):
    folded = [list(ax) for ax in unfolded]
    return [ax for group in folded for ax in group]

class Space(object):
    def __init__(self, axes, extent):
        self._axes = axes
        self._extent = extent

    @staticmethod
    def infer(X, axes):
        assert len(X.shape) == len(axes)
        return Space(axes, OrderedDict(zip(axes, X.shape)))

    def transform_axes(self, X, new_axes):
        new_extent = OrderedDict([
            (ax, self._extent.get(ax, 1))
            for ax in _fold_axes(new_axes)])
        new_space = Space(new_axes, new_extent)

        expanded_space = self._expand_axes(new_space.folded_axes)

        assert set(expanded_space.folded_axes) == set(new_space.folded_axes)

        X = expanded_space._fold(X)
        X = np.transpose(X, [expanded_space.folded_axes.index(d) for d in new_space.folded_axes])
        X = new_space._unfold(X)

        return X, new_space

    def _expand_axes(self, new_axes):
        shared_axes = [ax for ax in _fold_axes(new_axes) if ax in self.folded_axes]
        assert self.folded_axes == shared_axes

        new_extent = OrderedDict([
            (ax, self._extent.get(ax, 1))
            for ax in new_axes])
        new_space = Space(new_axes, new_extent)

        return new_space

    def replicate(self, X, **replicas):
        new_extent = OrderedDict([
            (ax, self._extent[ax]*replicas.get(ax, 1))
            for ax in _fold_axes(self.axes)])
        new_space = Space(self.axes, new_extent)

        X = self._fold(X)
        for ax, times in replicas.iteritems():
            X = np.concatenate([X]*times, axis=self.folded_axes.index(ax))
        assert X.size == new_space.size
        X = new_space._unfold(X)

        return X, new_space

    def _fold(self, X):
        return np.reshape(X, self.folded_shape)

    def _unfold(self, X):
        return np.reshape(X, self.shape)

    def set_extent(self, **extents):
        for ax,ex in extents.iteritems():
            self._extent[ax] = ex

    @property
    def size(self):
        return int(np.prod(self._extent.values()))

    @property
    def folded_shape(self):
        return self._extent.values()

    @property
    def shape(self):
        return [self._size_of_axis(ax) for ax in self._axes]

    def _size_of_axis(self, ax):
        return int(np.prod([v for k,v in self._extent.iteritems() if k in list(ax)]))

    @property
    def axes(self):
        return self._axes

    @property
    def folded_axes(self):
        return _fold_axes(self.axes)
