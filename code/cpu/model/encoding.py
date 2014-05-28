__author__ = 'mdenil, albandemiraj'

import numpy as np

import cpu.space
import cpu.model.layer

import generic.model.encoding


class DictionaryEncoding(generic.model.encoding.DictionaryEncoding, cpu.model.layer.Layer):
    def _fprop(self, X, meta):
        X = np.vstack([np.atleast_2d(x) for x in X])
        X_space = meta["space_below"]

        # Do we have a document provider or sentence provider below
        if ('b','s') in X_space._axes:
            # Masked the axis, now on we should continue normally until Doc Conv where we UnMask
            X_space=X_space.mask_axis(('b','s'), 'b')

        return X, X_space