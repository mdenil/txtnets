__author__ = 'mdenil'

import numpy as np
import unittest

from generic.space import *
from collections import OrderedDict

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
        self.assertEqual(Space(('a', 'b', 'c')).without_axes(to_remove).axes, ('c', ))
        self.assertEqual(Space((('a', 'b'), 'c')).without_axes(to_remove).axes, ('c',))
        self.assertEqual(Space((('a', 'b', 'd'), 'c')).without_axes(to_remove).axes, ('d', 'c'))
        self.assertEqual(Space((('a', 'c'), 'b', 'd')).without_axes(to_remove).axes, ('c', 'd'))
        self.assertEqual(Space((('x', 'b', 'c'),'q')).without_axes(to_remove).axes, (('x','c'),'q'))
        self.assertEqual(Space(('ab', 'c')).without_axes(to_remove).axes, ('ab', 'c'))

    def test_with_axes(self):
        extents = OrderedDict([('a', 2), ('b', 2), ('c', 3)])
        space = Space(['a', 'b', 'c'], extents)
        self.assertEqual(space.with_axes('d').axes, ('a', 'b', 'c', 'd'))
        self.assertEqual(space.with_axes('c').axes, ('a', 'b', 'c'))
        self.assertEqual(space.with_axes(('c', 'd')).axes, ('a', 'b', 'c', 'd'))

        self.assertEqual(space.with_axes('z').extents, OrderedDict([('a', 2), ('b', 2), ('c', 3), ('z', 1)]))

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

