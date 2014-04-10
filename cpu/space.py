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

    def transform(self, X, new_axes, **broadcast):
        new_extent = OrderedDict([
            (ax, self._extent.get(ax, 1))
            for ax in _fold_axes(new_axes)])
        new_space = Space(new_axes, new_extent)

        if set(self.folded_axes) <= set(new_space.folded_axes):
            expanded_axes = list(self.axes)
            expanded_extent = OrderedDict(self._extent)
            for ax in set(new_space.folded_axes) - set(self.folded_axes):
                expanded_axes.append(ax)
                expanded_extent[ax] = 1
            expanded_space = Space(expanded_axes, expanded_extent)
        else:
            expanded_space = self

        assert set(expanded_space.folded_axes) == set(new_space.folded_axes)

        X = expanded_space._fold(X)
        X = np.transpose(X, [expanded_space.folded_axes.index(d) for d in new_space.folded_axes])
        X = new_space._unfold(X)

        X, broadcast_space = new_space.broadcast(X, **broadcast)

        return X, broadcast_space

    def broadcast(self, X, **replicas):
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

    def without_axes(self, axes_to_drop):
        space = self.clone()

        axes_to_drop = list(axes_to_drop)

        for ax in axes_to_drop:
            assert ax in space._extent
            del space._extent[ax]

        new_axes = []
        for ax in space._axes:
            folded = _fold_axes(ax)
            folded = [f for f in folded if f not in axes_to_drop]
            new_axes.append(''.join(folded))

        space._axes = [ax for ax in new_axes if ax != '']

        assert set(space.folded_axes) == set(space._extent.keys())

        return space


    def _fold(self, X):
        return np.reshape(X, self.folded_shape)

    def _unfold(self, X):
        return np.reshape(X, self.shape)

    def with_extent(self, **extents):
        space = self.clone()
        for ax,ex in extents.iteritems():
            space._extent[ax] = ex
        return space

    def get_extents(self, axes):
        return [self.get_extent(ax) for ax in axes]

    def get_extent(self, ax):
        return int(np.prod([v for k,v in self._extent.iteritems() if k in list(ax)]))

    def clone(self):
        return Space(list(self._axes), OrderedDict(self._extent))

    @property
    def size(self):
        return int(np.prod(self._extent.values()))

    @property
    def folded_shape(self):
        return self._extent.values()

    @property
    def shape(self):
        return [self.get_extent(ax) for ax in self._axes]

    @property
    def axes(self):
        return self._axes

    @property
    def folded_axes(self):
        return _fold_axes(self.axes)

    def is_compatable_shape(self, X):
        return all(a == b for a,b in zip(X.shape, self.shape)) and len(X.shape) == len(self.shape)

    def __repr__(self):
        return "{}(axes={}, extent=[{}])".format(
            self.__class__.__name__,
            self.axes,
            ",".join("{}={}".format(k,v) for k,v in self._extent.iteritems()))
