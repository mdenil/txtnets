__author__ = 'mdenil'

import numpy as np

from . import tools
from cpu import space

class CSM(object):
    def __init__(self,
                 input_axes,
                 layers,
                 ):
        self.input_axes = input_axes
        self.layers = layers

    def fprop(self, X, lengths, data_space=None, num_layers=None, return_meta=False):
        if not data_space:
            data_space = space.Space.infer(X, self.input_axes)
        meta = {
            'lengths': lengths,
            'data_space': data_space,
        }

        self._Xs = []
        self._X_metas = []
        self._Ys = []
        self._Y_metas = []
        for layer_index, layer in enumerate(self.layers):
            if num_layers and layer_index == num_layers:
                break

            self._Xs.append(X)
            self._X_metas.append(meta)

            X, meta = layer.fprop(X, **meta)

            self._Ys.append(X)
            self._Y_metas.append(meta)

        if return_meta:
            return X, meta
        else:
            return X

    def bprop(self, delta):
        assert len(self.layers) == len(self._Y_metas)

        meta_above = None
        for layer, Y, meta in reversed(zip(self.layers, self._Ys, self._Y_metas)):
            if meta_above:
                delta, _ = meta_above['data_space'].transform(delta, meta['data_space'].axes)
            delta, meta_above = layer.bprop(Y, delta, **meta)

        return delta

    def grads(self, delta):
        assert len(self.layers) == len(self._Y_metas)
        assert len(self.layers) == len(self._X_metas)

        grads = []

        meta_above = None
        for layer, X, X_meta, Y, Y_meta in reversed(zip(self.layers, self._Xs, self._X_metas, self._Ys, self._Y_metas)):
            if meta_above:
                delta, _ = meta_above['data_space'].transform(delta, Y_meta['data_space'].axes)
            new_grads, _ = layer.grads(X, delta, **X_meta)
            delta, meta_above = layer.bprop(Y, delta, **Y_meta)

            # build list backwards because we're iterating backwards
            grads.extend(reversed(new_grads))

        return list(reversed(grads))

    def __repr__(self):
        return "\n".join([
            "CSM {",
            "\n".join(l.__repr__() for l in self.layers),
            "}"
        ])