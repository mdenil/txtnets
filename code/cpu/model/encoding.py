__author__ = 'albandemiraj'



__author__ = 'mdenil'

import numpy as np

import cpu.space
import cpu.model.layer

import generic.model.encoding


class DictionaryEncoding(generic.model.encoding.DictionaryEncoding, cpu.model.layer.Layer):
    def _fprop(self, X, meta):
        X = np.vstack([np.atleast_2d(x) for x in X])
        X_space = meta["space_below"]
        #X_space= cpu.space.CPUSpace.infer(X, ('b', 'w')) #We need the data provider to be able to inform the first level SPACE
        if ('b','s') in X_space._axes:                        #Do we have a document provider or sentence provider below
            X_space=X_space.mask_axis(('b','s'), 'b')   #Masked the axis, now on we should continue normally until Doc Conv where we UnMask

        return X, X_space