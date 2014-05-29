__author__ = 'mdenil'


# No! Numpy is not allowed in here >:|
# import numpy as np
import operator
#from collections import Dict


class Space(object):
    def __init__(self, axes, extents=None):
        self._axes = _canonical_axes_description(axes)

        if extents is None:
            extents = [(ax, 1) for ax in self.folded_axes]

        self._extents = dict(extents)

        if not set(self.folded_axes) == set(self._extents.keys()):
            raise ValueError("Cannot construct a space with axes={} and extents={}".format(self.axes, self.extents))

    # This interface must be implemented by subclasses.

    def fold(self, X):
        raise NotImplementedError

    def unfold(self, X):
        raise NotImplementedError

    def transpose(self, X, new_axes):
        raise NotImplementedError

    def broadcast(self, X, **replicas):
        raise NotImplementedError

    @classmethod
    def infer(cls, X, axes):
        axes = _canonical_axes_description(axes)

        if axes != _fold_axes(axes):
            raise ValueError("Cannot infer the shape of a space with unfolded axes (axes={})".format(axes))

        return cls(axes, zip(axes, X.shape))

    # @profile
    def transform(self, X, new_axes, **broadcast):
        self.check_compatible_shape(X)

        # short path for when there is no transposing
        if self.folded_axes == _fold_axes(new_axes):
            new_space = self.__class__(new_axes, self.extents)
            X = new_space.unfold(X)
            if len(broadcast) > 0:
                # broadcast any of the requested axes
                X, new_space = new_space.broadcast(X, **broadcast)
            return X, new_space

        new_axes = _canonical_axes_description(new_axes)
        new_folded_axes = _fold_axes(new_axes)

        new_space = self

        # add any new axes needed
        new_space = new_space.with_axes(new_folded_axes)

        # remove any size 1 axes that aren't in new_axes
        axes_to_drop = set(new_space.folded_axes) - set(new_folded_axes)
        if any(new_space.get_extent(ax) != 1 for ax in axes_to_drop):
            raise ValueError("You cannot drop an axis with extent != 1 using transform. (Tried to drop '{}' from {})"
                             ".".format(axes_to_drop, self))
        new_space = new_space.without_axes(axes_to_drop)
        # Reshape away the axes we just dropped. All of the dropped axes have size 1 so this doesn't
        # actually change the number of elements in the space, we're just updating metadata here.
        X = new_space.unfold(X)

        # permute the axes of X to align with the new order
        X, new_space = new_space.transpose(X, new_axes)

        if len(broadcast) > 0:
            # broadcast any of the requested axes
            X, new_space = new_space.broadcast(X, **broadcast)

        return X, new_space

    def add_axes(self, X, axes_to_add):
        self.check_compatible_shape(X)

        axes_to_add = _canonical_axes_description(axes_to_add)

        new_space = self.with_axes(axes_to_add)
        X = new_space.unfold(X)

        new_space.check_compatible_shape(X)

        return X, new_space

    def transposed(self, new_axes):
        """
        Rearranges the axes in this space into a new shape.

        This function can change the order and folding of axes, but cannot add or remove axes.

        :param new_axes:
        :return:
        """
        new_axes = _canonical_axes_description(new_axes)

        if set(self.folded_axes) != set(_fold_axes(new_axes)):
            raise ValueError("Axes incompatible for transposed. self.axes={}, new_axes={}".format(
                self.axes, new_axes))

        new_extents = dict((axis, self._extents[axis]) for axis in _fold_axes(new_axes))

        return self.__class__(new_axes, new_extents)

    # @profile
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
        expanded_extents = self.extents
        expanded_extents.update((ax, 1) for ax in new_axes)

        return self.__class__(expanded_axes, expanded_extents)

    def without_axes(self, axes_to_remove):
        """
        Creates a new space that does not include any of the specified axes.  You can't remove an unfolded axis, but you
        can remove an axis that is part of an axis that is unfolded in this space.  This means that you can't remove
        the axis ('a', 'b') from anything, even if this space has axes (('a', 'b'), 'c').  You _can_ remove 'a' from
        (('a', 'b'),'c') though, which will result in axes ('a', 'b').  You can also remove 'a' and 'b' which will
        result in axes ('c') which is probably what you wanted to do when you tried to remove ('a', 'b').

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

        contracted_extents = dict()
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

        return self.__class__(contracted_axes, contracted_extents)

    def with_extents(self, **extents_to_change):
        """
        Produce a new space with the specified extents set to new values.  Cannot add or remove axes
        with this method.  Does not modify folding.

        :param extents:
        :return:
        """
        new_extents = self.extents
        for ax, ex in extents_to_change.iteritems():
            if ax not in self.extents:
                raise ValueError("Tried to set the extent of axis {} in a space with axes={}".format(
                    ax, self.axes))

            new_extents[ax] = ex

        return self.__class__(self.axes, new_extents)

    def rename_axes(self, **renames):
        """
        You're not allowed to rename an axis that doesn't exist
        :param renames:
        :return:
        """
        for ax in renames.keys():
            if not ax in self._extents:
                raise ValueError("You're not allowed to rename an axis that doesn't exist.  Add it first (ax={})".format(ax))

        renamed_extents = dict()
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

        return self.__class__(renamed_axes, renamed_extents)

    def folded(self):
        return self.__class__(self.folded_axes, self.extents)

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
            if data_shape[d] != self_shape[s]:
                # If the shapes don't match, the spaces might still be compatible, as long as se == 1
                while s < len(self_shape) and self_shape[s] == 1:
                    s += 1
            if data_shape[d] != self_shape[s]:
                return False

            d += 1
            s += 1

        # the space is allowed to have extra trailing axes of size 1 that don't appear in X
        while s < len(self_shape) and self_shape[s] == 1:
            s += 1

        if d < len(data_shape) or s < len(self_shape):
            return False

        return True

    def check_compatible_shape(self, X):
        if not self.is_compatible_shape(X):
            raise ValueError("Incompatible shape X.shape={} and space={}".format(
                X.shape, self))

    @property
    def axes(self):
        return tuple(self._axes)

    @property
    def extents(self):
        return dict(self._extents)

    def get_extent(self, axis):
        """
        Get the extent of any single unfolded axis.  The needs to be unfolded (no nested tuples allowed).
        You can ask for an unfolded axis that doesn't exist in the space, eg you can ask for the extent
        of ('a', 'b') in a space with axes (('a','c'),'b').

        Axes not in the space will return an extent of 1.  This includes axes within an unfolded axis. i.e.
        you can ask for the extent of 'a' in ('b', 'c') and it will equal 1.
        """
        axis = _canonical_axes_description(axis)

        def prod(xs):
            # special case to make empty sequence = 1
            return reduce(operator.mul, xs) if xs else 1

        return prod([self._extents[ax] for ax in _protect_axis(axis) if ax in self._extents])

    def get_extents(self, axes):
        return map(self.get_extent, _canonical_axes_description(axes))

    @property
    def folded_axes(self):
        return _fold_axes(self._axes)

    @property
    def shape(self):
        return tuple(self.get_extent(ax) for ax in self._axes)

    @property
    def folded_shape(self):
        return tuple(self.get_extent(ax) for ax in _fold_axes(self._axes))

    def __repr__(self):
        return "{}(axes={}, extents=[{}])".format(
            self.__class__.__name__,
            self.axes,
            ",".join("{}={}".format(k, v) for k, v in self._extents.iteritems())
        )

def _canonical_axis_description(axis):
    if isinstance(axis, basestring):
        return axis
    return tuple(axis)


def _canonical_axes_description(axes):
    return tuple(map(_canonical_axis_description, _protect_axis(axes)))


def _fold_axes(axes):
    folded_axes = []
    for axis in axes:
        folded_axes.extend(_protect_axis(axis))
    return tuple(folded_axes)


def _protect_axis(axis):
    if isinstance(axis, basestring):
        return axis,
    else:
        return axis

