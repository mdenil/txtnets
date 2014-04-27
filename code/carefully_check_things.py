__author__ = 'mdenil'

import numpy as np
import scipy.optimize
import time

from cpu import conv

__author__ = 'mdenil'

import numpy as np
import itertools
from collections import OrderedDict
from collections import Iterable

def _fold_axis(axis):
    return tuple(axis.split(','))

def _fold_axes(axes):
    return tuple(itertools.chain(*(_fold_axis(ax) for ax in axes)))


class Space(object):
    def __init__(self, axes, extents):
        """
        :param extent: An OrderedDict (or sequence of pairs) mapping unfolded axes to unfolded extents.
        :return:
        """
        self._axes = tuple(axes)
        self._extents = OrderedDict(extents)

    @staticmethod
    def create_empty(axes):
        axes = tuple(axes)
        return Space(axes, ((ax,0) for ax in _fold_axes(axes)))

    @staticmethod
    def infer(X, axes):
        """
        Create a new space whose extents are infered from the shape of X.

        :param X: A matrix whose space is to be infered.
        :param axes: A sequence of axes.  Must be the same length as X.shape, and must be folded.
        :return:  A space describing X.
        """
        axes = tuple(axes)

        if axes != _fold_axes(axes):
            raise ValueError("Cannot infer the extent of a space with unfolded axes={}.".format(axes))

        if len(axes) != len(X.shape):
            raise ValueError("Cannot infer the extent of axes={} from X.shape={}".format(axes, X.shape))

        return Space(axes, zip(axes, X.shape))

    def folded(self):
        return Space(self.folded_axes, self.extents)

    def clone(self):
        return Space(self._axes, self._extents)

    def get_extent(self, axis):
        if not all(self.has_axis(ax) for ax in _fold_axis(axis)):
            raise ValueError("Space {} does not have axis={}".format(self, axis))

        return int(np.prod([ex for ax,ex in self._extents.iteritems() if ax in _fold_axis(axis)]))

    def has_axis(self, axis):
        return axis in self._extents


    def match_axis_order(self, other):
        """
        Add any dimensions in other but not in self.  Set size to 1.
        Drop any dimensions in self but not in other as long as they have size 1.
        (Any dimension in self but not in other that does not have size 1 is an error.)
        Check that self and other now have the same set of folded dimensions.
        Permute axes of self to match other.

        :param other:
        :return:
        """
        pass

    @property
    def extents(self):
        return self._extents

    @property
    def shape(self):
        return tuple(self.get_extent(ax) for ax in self._axes)

    @property
    def folded_shape(self):
        return self._extents.values()

    @property
    def size(self):
        return int(np.prod(self.shape))

    @property
    def axes(self):
        return self._axes

    @property
    def folded_axes(self):
        return self._extents.keys()

    @property
    def rank(self):
        return len(self._axes)

    @property
    def folded_rank(self):
        return len(self._extents)



    def __repr__(self):
        return "{}(axes={}, extent=[{}])".format(
            self.__class__.__name__,
            self._axes,
            ",".join("{}={}".format(k,v) for k,v in self._extents.iteritems()))



if __name__ == "__main__":
    np.random.seed(102312)
    np.set_printoptions(linewidth=250)


    X = np.random.uniform(size=(3,2,3))
    s = Space.infer(X, ['b','a','w'])
    print s
    print s.clone()

    s = Space.create_empty(['b,q', 'aw', 'c'])
    print s, s.shape, s.rank, s.folded_rank
    print s.folded(), s.folded().rank
    print s.folded(), s.folded().rank

    for ax in s.folded_axes:
        print ax, s.get_extent(ax)