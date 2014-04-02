__author__ = 'mdenil'

import numpy as np

from . import tools

class CSM(object):
    def __init__(self,
                 input_axes,
                 layers,
                 ):
        self.input_axes = input_axes
        self.layers = layers

    def fprop(self, X, meta=None, num_layers=None, return_meta=False):
        if not meta:
            meta = {}

        current_axes = self.input_axes
        for layer_index, layer in enumerate(self.layers):
            if num_layers and layer_index == num_layers:
                break

            # FIXME: have layers handle their own permutations.  Store current space in meta.
            X = tools.permute_axes(X, current_axes, layer.input_axes)
            X, meta = layer.fprop(X, **meta)
            current_axes = layer.output_axes

        if return_meta:
            return X, meta
        else:
            return X

    @property
    def output_axes(self):
        if len(self.layers) == 0:
            return None
        return self.layers[-1].output_axes

    def __repr__(self):
        return "\n".join([
            "CSM {",
            "\n".join(l.__repr__() for l in self.layers),
            "}"
        ])