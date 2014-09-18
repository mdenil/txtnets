__author__ = 'mdenil'

import numpy as np

import cpu.model.layer
import generic.model.repeat_layer

class RepeatLayer(generic.model.repeat_layer.RepeatLayer, cpu.model.layer.Layer):
    def _zeros_like(self, x):
        return np.zeros_like(x)