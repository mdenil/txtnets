__author__ = 'mdenil'

from gpu import space


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
            'space_below': space.GPUSpace.infer(self.X, axes=['b', 'w'])
        }

        return self.X, self.Y, meta