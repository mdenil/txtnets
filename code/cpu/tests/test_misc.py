__author__ = 'mdenil'

import numpy as np
import unittest

import cpu.space


class TestManualAxisFolding(unittest.TestCase):
    def test_manual_folding_first_axis(self):
        b, d, w = 10, 12, 14
        x = np.random.standard_normal(size=(b, d, w))
        space = cpu.space.CPUSpace.infer(x, ('b', 'd', 'w'))

        actual, space = space.transform(x, ('b2', 'b', 'd', 'w'))
        space = space.with_extents(
            b2=2,
            b=b//2)
        actual = space.unfold(actual)

        expected = np.concatenate([x[np.newaxis, :b//2], x[np.newaxis, b//2:]], axis=0)

        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(np.all(actual == expected))

    def test_manual_folding_second_axis(self):
        b, d, w = 10, 12, 14
        x = np.random.standard_normal(size=(b, d, w))
        space = cpu.space.CPUSpace.infer(x, ('b', 'd', 'w'))

        actual, space = space.transform(x, ('b', 'd2', 'd', 'w'))
        space = space.with_extents(
            d2=2,
            d=d//2)
        actual = space.unfold(actual)

        expected = np.concatenate([x[:, np.newaxis, :d//2], x[:, np.newaxis, d//2:]], axis=1)


    def test_manual_folding_third_axis(self):
        b, d, w = 10, 12, 14
        x = np.random.standard_normal(size=(b, d, w))
        space = cpu.space.CPUSpace.infer(x, ('b', 'd', 'w'))

        actual, space = space.transform(x, ('b', 'd', 'w2', 'w'))
        space = space.with_extents(
            w2=2,
            w=w//2)
        actual = space.unfold(actual)

        expected = np.concatenate([x[:, :, np.newaxis, :w//2], x[:, :, np.newaxis, w//2:]], axis=2)