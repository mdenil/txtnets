__author__ = 'mdenil'

import numpy as np
import scipy.optimize
import unittest

import cpu.optimize.data_provider


class PaddedParallelSequenceMinibatchProvider(unittest.TestCase):
    def setUp(self):
        self.X1 = [[x, 2*x] for x in range(10)]
        self.X2 = [[x, 2*x+1] for x in range(10)]

    def test_syncronized(self):
        provider = cpu.optimize.data_provider.PaddedParallelSequenceMinibatchProvider(
            X1=list(self.X1),
            X2=list(self.X1),
            batch_size=len(self.X1) / 2,
            padding="XX")

        # iterate a few times to make sure the batches don't get out of sync
        for _ in xrange(10):
            X1, meta1, X2, meta2 = provider.next_batch()
            self.assertEqual(X1, X2)

            print X1, X2