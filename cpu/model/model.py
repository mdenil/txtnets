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

    def fprop(self, X, meta, num_layers=None, return_state=False, return_meta=False):
        if 'space_below' not in meta:
            meta['space_below'] = space.Space.infer(X, self.input_axes)

        self._Xs = []
        self._X_metas = []
        self._Ys = []
        self._Y_metas = []

        fprop_state = []

        for layer_index, layer in enumerate(self.layers):
            if num_layers and layer_index == num_layers:
                break

            self._Xs.append(X)
            self._X_metas.append(dict(meta))

            if 'space_above' in meta:
                meta['space_below'] = meta['space_above']
                del meta['space_above']

            if not meta['space_below'].is_compatable_shape(X):
                raise ValueError("Layer of type '{}' produced data with shape {}, but claims it has shape {}.".format(
                    type(self.layers[layer_index-1]),
                    X.shape,
                    meta['space_below'].shape))

            ret = layer.fprop(X, meta=dict(meta))

            if len(ret) == 3:
                X, meta, layer_fprop_state = ret
            else:
                X, meta = ret
                layer_fprop_state = {}

            if 'space_above' not in meta:
                meta['space_above'] = meta['space_below']

            self._Ys.append(X)
            self._Y_metas.append(dict(meta))

            fprop_state.append(layer_fprop_state)

        ret = [X]


        if return_meta:
            ret.append(meta)

        if return_state:
            ret.append(fprop_state)

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def bprop(self, delta, fprop_state):
        assert len(self.layers) == len(self._Y_metas)

        meta_above = None
        for layer, layer_fprop_state, Y, meta in reversed(zip(self.layers, fprop_state, self._Ys, self._Y_metas)):
            if meta_above:
                delta, _ = meta_above['space_below'].transform(delta, meta['space_above'].axes)
            delta, meta_above = layer.bprop(Y, delta, fprop_state=layer_fprop_state, meta=meta)

        return delta

    # TOOD: re-write this correctly:
    # def grads(self, delta):
    #     assert len(self.layers) == len(self._Y_metas)
    #     assert len(self.layers) == len(self._X_metas)
    #
    #     grads = []
    #
    #     meta_above = None
    #     for layer, X, X_meta, Y, Y_meta in reversed(zip(self.layers, self._Xs, self._X_metas, self._Ys, self._Y_metas)):
    #         if meta_above:
    #             delta, _ = meta_above['data_space'].transform(delta, Y_meta['data_space'].axes)
    #         new_grads, _ = layer.grads(X, delta, meta=X_meta)
    #         delta, meta_above = layer.bprop(Y, delta, meta=Y_meta)
    #
    #         # build list backwards because we're iterating backwards
    #         grads.extend(reversed(new_grads))
    #
    #     return list(reversed(grads))

    def __repr__(self):
        return "\n".join([
            "CSM {",
            "\n".join(l.__repr__() for l in self.layers),
            "}"
        ])