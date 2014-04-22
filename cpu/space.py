__author__ = 'mdenil'

import numpy as np

from generic.space import Space
from generic.space import _canonical_axes_description
from generic.space import _fold_axes


class CPUSpace(Space):
    def __init__(self, axes, extents=None):
        super(CPUSpace, self).__init__(axes, extents)

    def _fold(self, X):
        return np.reshape(X, self.folded_shape)

    def _unfold(self, X):
        return np.reshape(X, self.shape)

    def transpose(self, X, new_axes):
        self.check_compatible_shape(X)

        new_axes = _canonical_axes_description(new_axes)

        X = self._fold(X)
        X = np.transpose(X, [self.folded_axes.index(axis) for axis in _fold_axes(new_axes)])
        new_space = self.transposed(new_axes)
        X = new_space._unfold(X)

        new_space.check_compatible_shape(X)

        return X, new_space

    def add_axes(self, X, axes_to_add):
        self.check_compatible_shape(X)

        axes_to_add = _canonical_axes_description(axes_to_add)

        new_space = self.with_axes(axes_to_add)

        # some of the axes_to_add may already exist, that's okay.
        new_axes = set(new_space.folded_axes) - set(self.folded_axes)
        for axis in new_axes:
            X = X[..., np.newaxis]

        new_space.check_compatible_shape(X)

        return X, new_space

    def broadcast(self, X, **replicas):
        self.check_compatible_shape(X)

        new_space = self.with_axes(replicas.keys())

        X = new_space._fold(X)
        for axis, times in replicas.iteritems():
            X = np.concatenate([X]*times, axis=new_space.folded_axes.index(axis))
            new_space = new_space.with_extents(**{axis: new_space.get_extent(axis) * times})
        X = new_space._unfold(X)

        return X, new_space

