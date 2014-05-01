__author__ = 'mdenil'

import numpy as np

import random
from collections import OrderedDict

import cpu.space

class LabelledSequenceMinibatchProvider(object):
    def __init__(self, X, Y, batch_size, padding, shuffle=True, fixed_length=False):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.padding = padding
        self.fixed_length = fixed_length
        self.shuffle = shuffle

        self._batch_index = -1
        self.batches_per_epoch = len(X) / batch_size

    def next_batch(self):
        self._prepare_for_next_batch()

        batch_start = self._batch_index * self.batch_size
        batch_end = batch_start + self.batch_size

        X_batch = self.X[batch_start:batch_end]
        Y_batch = self.Y[batch_start:batch_end]

        Y_batch = np.equal.outer(Y_batch, np.arange(np.max(Y_batch)+1)).astype(np.float)

        lengths_batch = np.asarray(map(len, X_batch))

        if self.fixed_length:
            max_length_batch = self.fixed_length
            lengths_batch = np.minimum(lengths_batch, self.fixed_length)
        else:
            max_length_batch = int(lengths_batch.max())

        X_batch = [self._pad_or_truncate(x, max_length_batch) for x in X_batch]

        meta = {
            'lengths': lengths_batch,
            'space_below': cpu.space.CPUSpace(
                axes=['b', 'w'],
                extents=OrderedDict([('b', len(X_batch)), ('w', max_length_batch)]))
        }

        return X_batch, Y_batch, meta

    def _prepare_for_next_batch(self):
        self._batch_index = (self._batch_index + 1) % self.batches_per_epoch

        if self._batch_index == 0 and self.shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        combined = zip(self.X, self.Y)
        random.shuffle(combined)
        self.X, self.Y = map(list, zip(*combined))

    def _pad_or_truncate(self, x, max_length):
        if max_length > len(x):
            return x + [self.padding] * (max_length - len(x))
        else:
            return x[:max_length]


class LabelledSequenceBatchProvider(LabelledSequenceMinibatchProvider):
    def __init__(self, X, Y, padding):
        super(LabelledSequenceBatchProvider, self).__init__(
            X, Y, len(X), padding, shuffle=False, fixed_length=False)