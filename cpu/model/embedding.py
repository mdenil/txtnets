__author__ = 'mdenil'

import numpy as np

class WordEmbedding(object):
    def __init__(self,
                 dimension,
                 vocabulary_size,
                 ):

        self.dimension = dimension
        self.vocabulary_size = vocabulary_size

        self.E = 0.05 * np.random.standard_normal(size=(self.vocabulary_size, self.dimension))

        self.input_axes = ['b', 'w']
        self.output_axes = ['b', 'w', 'd']

    def fprop(self, X, **meta):
        b, w = X.shape

        # ravel() unwinds in C order, (row major).  The result is each sentence appears consecutavely.
        # There is one word per row, and sentence are stacked vertically.
        X = self.E[X.ravel()]
        X = np.reshape(X, (b, w, self.dimension))

        return X, meta

    def __repr__(self):
        return "{}(dim={}, vocab_size={})".format(
            self.__class__.__name__,
            self.dimension,
            self.vocabulary_size)
