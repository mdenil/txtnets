__author__ = 'mdenil'

import numpy as np

import itertools

from . import tools
from cpu import space
from cpu.model import layer

class CSM(layer.Layer):
    def __init__(self,
                 layers,
                 ):
        self.layers = layers

    def fprop(self, X, meta, num_layers=None, return_state=False, input_axes=None):
        if 'space_below' not in meta:
            assert input_axes is not None
            meta['space_below'] = space.Space.infer(X, input_axes)

        layer_fprop_states = []

        for layer_index, layer in enumerate(self.layers):
            if num_layers and layer_index == num_layers:
                break

            if layer_index > 0:
                meta['space_below'] = meta['space_above']
                del meta['space_above']

            if not meta['space_below'].is_compatable_shape(X):
                raise ValueError("Layer of type '{}' produced data with shape {}, but claims it has shape {}.".format(
                    type(self.layers[layer_index-1]),
                    X.shape,
                    meta['space_below'].shape))

            X, meta, layer_fprop_state = _ensure_layer_fprop_state(layer.fprop(X, meta=dict(meta)))

            if 'space_above' not in meta:
                meta['space_above'] = meta['space_below']

            layer_fprop_states.append(layer_fprop_state)

        fprop_state = {
            'layer_fprop_states': layer_fprop_states,
        }

        if not return_state:
            return X
        else:
            return X, meta, fprop_state


    def bprop(self, delta, meta, fprop_state, return_state=False):
        layer_fprop_states = fprop_state['layer_fprop_states']

        assert len(self.layers) == len(layer_fprop_states)

        for layer_index, layer, layer_fprop_state in reversed(zip(itertools.count(), self.layers, layer_fprop_states)):
            # The space below the layer above is the space above the current layer, we dont know what the space below
            # this layer is yet.
            if layer_index < len(self.layers) - 1:
                meta['space_above'] = meta['space_below']
            del meta['space_below']

            # layers should _not_ assume Y and delta are in the same space.  Y will be in the space they produced
            # (and they can store this space info in layer_fprop_state if they want),
            # delta will be described by meta['space_above']

            delta, meta = layer.bprop(delta, fprop_state=layer_fprop_state, meta=dict(meta))

        if not return_state:
            return delta
        else:
            return delta, meta

    def grads(self, delta, meta, fprop_state):
        layer_fprop_states = fprop_state['layer_fprop_states']

        assert len(self.layers) == len(layer_fprop_states)

        grads = []

        for layer_index, layer, layer_fprop_state in reversed(zip(itertools.count(), self.layers, layer_fprop_states)):


            if layer_index < len(self.layers) - 1:
                meta['space_above'] = meta['space_below']
            del meta['space_below']

            new_grads = layer.grads(delta, meta=dict(meta), fprop_state=layer_fprop_state)
            delta, meta = layer.bprop(delta, meta=dict(meta), fprop_state=layer_fprop_state)

            # build list backwards because we're iterating backwards
            grads.extend(reversed(new_grads))

        return list(reversed(grads))

    def params(self):
        params = []
        for layer in self.layers:
            params.extend(layer.params())
        return params

    def __repr__(self):
        return "\n".join([
            "CSM {",
            "\n".join(l.__repr__() for l in self.layers),
            "}"
        ])



def _ensure_layer_fprop_state(ret):
    if len(ret) == 3:
        X, meta, layer_fprop_state = ret
    else:
        X, meta = ret
        layer_fprop_state = {}

    return X, meta, layer_fprop_state