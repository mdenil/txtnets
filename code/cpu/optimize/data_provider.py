__author__ = 'mdenil'

import numpy as np
import random
from collections import OrderedDict

import generic.optimize.data_provider

from cpu import space


class LabelledSequenceMinibatchProvider(
        generic.optimize.data_provider.LabelledSequenceMinibatchProvider):
    pass


class LabelledSequenceBatchProvider(
        generic.optimize.data_provider.LabelledSequenceBatchProvider):
    pass


class SequenceMinibatchProvider(
        generic.optimize.data_provider.SequenceMinibatchProvider):
    pass


class PaddedParallelSequenceMinibatchProvider(
        generic.optimize.data_provider.PaddedParallelSequenceMinibatchProvider):
    pass


class LabelledDocumentMinibatchProvider(
        generic.optimize.data_provider.LabelledDocumentMinibatchProvider):
    pass


class ShardedLabelledDocumentMinibatchProvider(
        generic.optimize.data_provider.ShardedLabelledDocumentMinibatchProvider):
    pass

class TransformedLabelledDataProvider(
        generic.optimize.data_provider.TransformedLabelledDataProvider):
    pass


class MinibatchDataProvider(object):
    def __init__(self, X, Y, lengths, batch_size):
        self.X = X
        self.Y = Y
        self.lengths = lengths
        self.batch_size = batch_size

        self._batch_index = -1 # will be incremented to 0 when next_batch is called
        self.batches_per_epoch = int(self.X.shape[0] / self.batch_size)

    def next_batch(self):
        self._prepare_for_next_batch()

        batch_start = self._batch_index * self.batch_size
        batch_end = batch_start + self.batch_size

        X_batch = self.X[batch_start:batch_end]
        Y_batch = self.Y[batch_start:batch_end]
        lengths_batch = self.lengths[batch_start:batch_end]

        meta = {
            'lengths': lengths_batch,
            'space_below': space.CPUSpace.infer(X_batch, axes=['b', 'w']),
            }

        return X_batch, Y_batch, meta

    def _prepare_for_next_batch(self):
        self._batch_index = (self._batch_index + 1) % self.batches_per_epoch

        if self._batch_index == 0:
            self._shuffle_data()

    def _shuffle_data(self):
        perm = np.random.permutation(self.X.shape[0])
        self.X = self.X[perm]
        self.Y = self.Y[perm]
        self.lengths = self.lengths[perm]



class BatchDataProvider(object):
    def __init__(self, X, Y, lengths):
        self.X = X
        self.Y = Y
        self.lengths = lengths

        self.batch_size = X.shape[0]
        self.batches_per_epoch = 1

    def next_batch(self):
        meta = {
            'lengths': self.lengths,
            'space_below': space.CPUSpace.infer(self.X, axes=['b', 'w'])
        }

        return self.X, self.Y, meta


class PaddedSequenceMinibatchProvider(object):
    def __init__(self, X, batch_size, padding, shuffle=True):
        self.X = X
        self.batch_size = batch_size
        self.padding = padding
        self.shuffle = shuffle

        self._batch_index = -1 # will be incremeted to 0 when next_batch is called
        self.batches_per_epoch = len(X) / self.batch_size

    def next_batch(self):
        self._prepare_for_next_batch()

        batch_start = self._batch_index * self.batch_size
        batch_end = batch_start + self.batch_size

        X_batch = self.X[batch_start:batch_end]

        lengths_batch = np.asarray(map(len, X_batch))
        max_length_batch = lengths_batch.max()

        # X_batch = np.vstack([np.atleast_2d(self._add_padding(x, max_length_batch)) for x in X_batch])
        X_batch = [self._add_padding(x, max_length_batch) for x in X_batch]

        # meta = {
        #     'lengths': lengths_batch,
        #     'space_below': space.CPUSpace.infer(X_batch, axes=['b', 'w'])
        # }

        meta = {
            'lengths': lengths_batch,
            'space_below': space.CPUSpace(
                axes=['b', 'w'],
                extents=OrderedDict([('b', len(X_batch)), ('w', max_length_batch)]))
        }

        return X_batch, meta

    def _add_padding(self, x, length):
        return x + [self.padding] * (length - len(x))

    def _prepare_for_next_batch(self):
        self._batch_index = (self._batch_index + 1) % self.batches_per_epoch

        if self._batch_index == 0 and self.shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        random.shuffle(self.X)


