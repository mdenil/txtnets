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
        if not self.is_compatable_shape(X):
            raise ValueError("Matrix of shape {} not compatable with {}".format(X.shape, self))

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
        elif set(new_space.folded_axes) <= set(self.folded_axes):
            expanded_space = self

            extra_axes = set(self.folded_axes) - set(new_space.folded_axes)
            for ax in extra_axes:
                if expanded_space.get_extent(ax) == 1:
                    expanded_space = expanded_space.without_axes(ax)
                else:
                    raise ValueError("Can't transform {} to {}".format(self, new_space))
        else:
            expanded_space = self

        assert set(expanded_space.folded_axes) == set(new_space.folded_axes)

        X = expanded_space._fold(X)
        X = np.transpose(X, [expanded_space.folded_axes.index(d) for d in new_space.folded_axes])
        X = new_space._unfold(X)

        X, broadcast_space = new_space.broadcast(X, **broadcast)

        return X, broadcast_space

    def broadcast(self, X, **replicas):
        if not self.is_compatable_shape(X):
            raise ValueError("Matrix of shape {} not compatable with {}".format(X.shape, self))

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
        # return all(a == b for a,b in zip(X.shape, self.shape)) and len(X.shape) == len(self.shape)

        # Compatable is not just having the same shape.  The space is also allowed to include extra dimensions as long
        # as they have size 1, so a matrix of shape (10, 4) is compatable with a space of shape (10, 1, 4) or (1, 10, 4),
        # etc, but is _not_ compatable with a space of shape (4, 10).  However, a space must describe every dimension of
        # the data, so a space with shape (10, 4) is _not_ compatable with a matrix of shape (10, 1, 4) even though the
        # reverse is true.

        data_extents = X.shape
        self_extents = self.shape

        d = 0
        s = 0
        while d < len(data_extents) and s < len(self_extents):
            de = data_extents[d]
            se = self_extents[s]

            if de != se:
                while se == 1 and s < len(self_extents):
                    s += 1
                    se = self_extents[s]
            if de != se:
                return False

            d += 1
            s += 1

        if d < len(data_extents) or s < len(self_extents):
            return False

        return True

    def __repr__(self):
        return "{}(axes={}, extent=[{}])".format(
            self.__class__.__name__,
            self.axes,
            ",".join("{}={}".format(k,v) for k,v in self._extent.iteritems()))
