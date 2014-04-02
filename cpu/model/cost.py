__author__ = 'mdenil'

import numpy as np

class CrossEntropy(object):
    def __init__(self):
        pass

    def fprop(self, Y, Y_true, **meta):
        # really bad things happen if Y_true is a bool array

        out = - np.sum(Y_true * np.log(Y))

        return out, meta


    def bprop(self, Y, Y_true, **meta):
        back = Y - Y_true

        return back, meta

    def __repr__(self):
        return "{}()".format(
            self.__class__.__name__)