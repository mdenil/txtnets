__author__ = 'mdenil'

import numpy as np
import functools
import itertools
from collections import OrderedDict


class Space(object):
    def __init__(self, axes, extents=None):
        self._axes = _canonical_axes_description(axes)

        if extents is None:
            extents = [(ax, 1) for ax in self.folded_axes]

        self._extents = OrderedDict(extents)

        if not self.folded_axes == tuple(self._extents.keys()):
            raise ValueError("Cannot construct a space with axes={} and extents={}".format(self.axes, self.extents))

    @staticmethod
    def infer(X, axes):
        axes = _canonical_axes_description(axes)

        if axes != _fold_axes(axes):
            raise ValueError("Cannot infer the shape of a space with unfolded axes (axes={})".format(axes))

        return Space(axes, zip(axes, X.shape))

    def transposed(self, new_axes):
        """
        Rearranges the axes in this space into a new shape.

        This function can change the order and folding of axes, but cannot add or remove axes.

        :param new_axes:
        :return:
        """
        new_axes = _canonical_axes_description(new_axes)

        if self.folded_axes != _fold_axes(new_axes):
            raise ValueError("Axes incompatible for transpose. self.axes={}, new_axes={}".format(
                self.axes, new_axes))

        new_extents = OrderedDict((axis, self._extents[axis]) for axis in _fold_axes(new_axes))

        return Space(new_axes, new_extents)

    def with_axes(self, axes):
        """
        Creates a new space that includes any of the specified axes.  You can't add an unfolded axis, but you can add
        an axis that already exists (which does nothing).  Newly added axes will have extent 1 and will appear to the
        right of any existing axes in the order they are specified.

        :param axes:
        :return:
        """
        axes = _canonical_axes_description(axes)

        if axes != _fold_axes(axes):
            raise ValueError("You can only add folded axes to a space (you tried to add {}).".format(axes))

        new_axes = tuple(ax for ax in axes if ax not in self.folded_axes)

        expanded_axes = self.axes + new_axes
        # using .extents makes a copy, so this doesn't change self._extents
        expanded_extents = self.extents.update((ax, 1) for ax in new_axes)

        return Space(expanded_axes, expanded_extents)

    def without_axes(self, *axes_to_remove):
        """
        Creates a new space that does not include any of the specified axes.  You can't remove an unfolded axis, but you
        can remove an axis that is part of an axis that is unfolded in this space.  This means that you can't remove
        the axis ('a', 'b') from anything, even if this space has axes (('a', 'b'), 'c').  You _can_ remove 'a' from
        (('a', 'b'),'c') though, which will result in axes ('a', 'b').  You can also remove 'a' and 'b' which will
        result in axes ('c') which is probaby what you wanted to do when you tried to remove ('a', 'b').

        Removing all but one axis in a folded axis will unfold the axis.  This means that removing 'a' from (('a','b'),'c')
        will result in ('b',c') and not in (('b',),'c').

        Removing axes is required to be commutative, so removing 'a' then 'b' must always be the same as removing 'b'
        then 'a'.

        You can also remove axes that are not present, which does nothing.  i.e. you can remove 'c' from ('a', 'b')
        which results in axes ('a', 'b').

        :param axes_to_remove:
        :return:
        """
        axes_to_remove = _canonical_axes_description(axes_to_remove)

        if axes_to_remove != _fold_axes(axes_to_remove):
            raise ValueError("You cannot remove an unfolded axis from a space.  Remove each axis individually instead.")

        contracted_extents = OrderedDict()
        for ax in self._extents:
            if ax not in axes_to_remove:
                contracted_extents[ax] = self._extents[ax]

        contracted_axes = []
        for axis in self._axes:
            # if the axis is unfolded
            if isinstance(axis, tuple):
                # remove axes from the unfolded part
                contracted_axis = tuple(ax for ax in axis if ax not in axes_to_remove)

                if len(contracted_axis) == 1:
                    # if removing axes has left only one element in the unfolded axis then it is no longer unfolded
                    contracted_axes.append(contracted_axis[0])

                elif len(contracted_axis) > 1:
                    # if there are still multiple elements in the unfolded axis then it remains unfolded
                    contracted_axes.append(contracted_axis)

                else:
                    # If we've removed all the elements of the unfolded axis then the whole axis is removed
                    pass

            elif axis not in axes_to_remove:
                contracted_axes.append(axis)

        contracted_axes = tuple(contracted_axes)

        return Space(contracted_axes, contracted_extents)


    def with_extents(self, **new_extents):
        """
        Produce a new space with the specified extents set to new values.  Cannot add or remove axes
        with this method.  Does not modify folding.

        :param extents:
        :return:
        """
        new_extents = self.extents
        for ax, ex in new_extents.iteritems():
            if ax not in new_extents:
                raise ValueError("Tried to set the extent of axis {} in a space with axes={}".format(
                    ax, self.axes))

            new_extents[ax] = ex

        return Space(self.axes, new_extents)

    def rename_axes(self, **renames):
        """
        You're not allowed to rename an axis that doesn't exist
        :param renames:
        :return:
        """
        for ax in renames.keys():
            if not ax in self._extents:
                raise ValueError("You're not allowed to rename an axis that doesn't exist.  Add it first (ax={})".format(ax))

        renamed_extents = OrderedDict()
        for ax, ex in self._extents.iteritems():
            if ax in renames:
                ax = renames[ax]
            renamed_extents[ax] = ex

        renamed_axes = []
        for axis in self._axes:
            # if the axis is unfolded
            if isinstance(axis, tuple):
                renamed_axes.append(tuple(renames.get(ax, ax) for ax in axis))
            else:
                renamed_axes.append(renames.get(axis, axis))
        renamed_axes = tuple(renamed_axes)

        return Space(renamed_axes, renamed_extents)

    def folded(self):
        return Space(self.folded_axes, self.extents)

    def is_compatible_shape(self, X):
        """
        Compatible is not just having the same shape.  The space is also allowed to include extra dimensions as long
        as they have size 1, so a matrix of shape (10, 4) is compatible with a space of shape (10, 1, 4) or (1, 10, 4),
        etc, but is _not_ compatible with a space of shape (4, 10).  However, a space must describe every dimension of
        the data, so a space with shape (10, 4) is _not_ compatible with a matrix of shape (10, 1, 4) even though the
        reverse is true.
        """

        data_shape = X.shape
        self_shape = self.shape

        d = 0
        s = 0
        while d < len(data_shape) and s < len(self_shape):
            de = data_shape[d]
            se = self_shape[s]

            if de != se:
                while se == 1 and s < len(self_shape):
                    s += 1
                    se = self_shape[s]
            if de != se:
                return False

            d += 1
            s += 1

        if d < len(data_shape) or s < len(self_shape):
            return False

        return True

    def check_compatible_shape(self, X):
        if not self.is_compatible_shape(X):
            raise ValueError("Incompatible shape X.shape={} and space.shape={}".format(
                X.shape, self.shape))

    @property
    def axes(self):
        return tuple(self._axes)

    @property
    def extents(self):
        return OrderedDict(self._extents)

    def get_extent(self, axis):
        """
        Get the extent of any single unfolded axis.  The needs to be unfolded (no nested tuples allowed).
        You can ask for an unfolded axis that doesn't exist in the space, as long as all of the component axes are
        part of the folded space.  Ie you can ask for the extent of ('a', 'b') in a space with axes (('a','c'),'b')
        but you cannot ask for the extent of 'q'.
        """
        axis = _canonical_axes_description(axis)

        return int(np.prod([self._extents[ax] for ax in _protect_axis(axis)]))

    @property
    def folded_axes(self):
        return _fold_axes(self._axes)

    @property
    def shape(self):
        return tuple(self.get_extent(ax) for ax in self._axes)

    @property
    def folded_shape(self):
        return tuple(self._extents.values())

    def __repr__(self):
        return "{}(axes={}, extents=[{}])".format(
            self.__class__.__name__,
            self.axes,
            ",".join("{}={}".format(k, v) for k, v in self._extents.iteritems()))


def _canonical_axis_description(axis):
    if isinstance(axis, basestring):
        return axis
    return tuple(axis)


def _canonical_axes_description(axes):
    return tuple(map(_canonical_axis_description, _protect_axis(axes)))


def _fold_axes(axes):
    flat_axes = []
    for axis in axes:
        flat_axes.extend(_protect_axis(axis))
    return tuple(flat_axes)


def _protect_axis(x):
    if isinstance(x, basestring):
        return x,
    else:
        return x


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
            new_space = new_space.with_extents(axis=new_space.get_extent(axis) * times)
            new_space.check_compatible_shape(X)

        return X, new_space





import unittest


class TestSpace(unittest.TestCase):

    def test_infer(self):
        X = np.zeros((3,5,4))
        space = Space.infer(X, ('a', 'b', 'c'))

        self.assertEqual(space.shape, X.shape)
        self.assertEqual(space.axes, ('a', 'b', 'c'))

        space = Space.infer(X, ('ab', 'c', 'd'))
        self.assertEqual(space.shape, X.shape)
        self.assertEqual(space.axes, ('ab', 'c', 'd'))

        with self.assertRaises(ValueError):
            X = np.zeros((3,4))
            space = Space.infer(X, (('a', 'b'),'c'))

    def test_rename_axes(self):
        space = Space((('a','b'),'c'))
        self.assertEqual(space.rename_axes(a='q').axes, (('q','b'),'c'))

    def test_without_axes(self):
        to_remove = ('a', 'b')
        self.assertEqual(Space(('a', 'b', 'c')).without_axes(*to_remove).axes, ('c', ))
        self.assertEqual(Space((('a', 'b'), 'c')).without_axes(*to_remove).axes, ('c',))
        self.assertEqual(Space((('a', 'b', 'd'), 'c')).without_axes(*to_remove).axes, ('d', 'c'))
        self.assertEqual(Space((('a', 'c'), 'b', 'd')).without_axes(*to_remove).axes, ('c', 'd'))
        self.assertEqual(Space((('x', 'b', 'c'),'q')).without_axes(*to_remove).axes, (('x','c'),'q'))
        self.assertEqual(Space(('ab', 'c')).without_axes(*to_remove).axes, ('ab', 'c'))

    def test_with_axes(self):
        space = Space(['a', 'b', 'c'])
        self.assertEqual(space.with_axes('d').axes, ('a', 'b', 'c', 'd'))
        self.assertEqual(space.with_axes('c').axes, ('a', 'b', 'c'))
        self.assertEqual(space.with_axes(('c', 'd')).axes, ('a', 'b', 'c', 'd'))

        for ax in ['a', 'b', 'c', 'd']:
            self.assertEqual(tuple(space.with_axes(ax).extents.keys()), space.with_axes(ax).folded_axes)

        with self.assertRaises(ValueError):
            space.with_axes(('a',('b',)))

    def test_folded(self):
        axes = (('a', 'b'), 'c', ('d', 'e', 'f',), 'g')
        extents = OrderedDict([('a', 1), ('b', 2), ('c', 3), ('d', 4), ('e', 5), ('f', 6), ('g', 7)])
        space = Space(axes, extents)

        self.assertEqual(space.folded_shape, space.folded().shape)
        self.assertEqual(space.folded_axes, space.folded().axes)

    def test_copy_extents(self):
        axes = ('a', 'b')
        extents = OrderedDict([('a', 2), ('b', 3)])
        space = Space(axes, extents)

        del extents['a']
        self.assertFalse('a' in extents)
        self.assertTrue('a' in space.extents)

    def test_axes_immutable(self):
        axes = (['a', 'b'],)
        extents = OrderedDict([('a', 2), ('b', 3)])
        space = Space(axes, extents)

        expected_axes = (('a', 'b'),)
        self.assertEqual(space.axes, expected_axes)

        axes[0][0] = 'c'
        self.assertEqual(axes, (['c', 'b'],))
        self.assertEqual(space.axes, expected_axes)

    def test_folded_axes(self):
        extents = OrderedDict([('a', 2), ('b', 3)])

        axes = (['a', 'b'],)
        space = Space(axes, extents)
        self.assertEqual(space.axes, (('a','b'),))
        self.assertEqual(space.folded_axes, ('a', 'b'))

        axes = ('a', 'b')
        space = Space(axes, extents)
        self.assertEqual(space.axes, space.folded_axes)

    def test_folded_shape(self):
        extents = OrderedDict([('a', 2), ('b', 3)])
        space = Space(extents.keys(), extents)
        self.assertEqual(space.shape, space.folded_shape)

        extents = OrderedDict([('a', 2), ('b', 3)])
        space = Space((('a', 'b',),), extents)
        self.assertNotEqual(space.shape, space.folded_shape)
        self.assertEqual(space.folded_shape, (2,3))

    def test_shape(self):
        extents = OrderedDict([('a', 2), ('b', 3), ('c', 7)])

        space = Space((('a','b'),'c'), extents)
        self.assertEqual(space.shape, (6, 7))

        space = Space(('a', ('b', 'c')), extents)
        self.assertEqual(space.shape, (2, 21))

    def test_get_extent(self):
        extents = OrderedDict([('a', 2), ('b', 3), ('c', 7)])
        axes = ('a', 'b', 'c')
        space = Space(axes, extents)

        self.assertEqual(space.get_extent('b'), 3)
        self.assertEqual(space.get_extent(('b',)), 3)
        self.assertEqual(space.get_extent(['a', 'b']), 6)

    def _canonical_axes_description(self):
        self.assertEqual(_canonical_axes_description([['a','b'],'c']), (('a','b'),'c'))
        self.assertEqual(_canonical_axes_description('a'), ('a',))
        self.assertEqual(_canonical_axes_description([['a', 'b']]), (('a','b'),))
        self.assertEqual(_canonical_axes_description('a'), ('a',))

