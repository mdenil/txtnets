__author__ = 'albandemiraj'

from cpu.space import CPUSpace
import numpy as np
import cpu.model.layer
from cpu.model.model import CSM


class Dropout(cpu.model.layer.Layer):
    def __init__(self, axes, dropout_rate):
        self.axes = axes
        self.dropout_rate = dropout_rate


    def fprop(self, X, meta):
        X = X
        X_space = meta['space_below']

        mask = np.random.uniform(size=(X_space.get_extents(self.axes)))<self.dropout_rate
        mask_space = CPUSpace.infer(mask, self.axes)

        extents = X_space.extents
        for ax in self.axes:
            extents.pop(ax)

        mask, mask_space = mask_space.transform(mask, X_space.axes, **extents)

        Y = X * mask

        meta['space_above'] = X_space

        fprop_state = {
            'mask' : mask,
            'mask_space' : mask_space,
            'X_space' : X_space
            }

        return Y, meta, fprop_state

    def bprop(self, delta, meta, fprop_state):
        delta_space = meta['space_above']
        mask = fprop_state['mask']
        mask_space = fprop_state['mask_space']

        mask, mask_space = mask_space.transform(mask, delta_space.axes)

        out = delta * mask

        space_below = fprop_state['X_space']
        meta['space_below'] = space_below

        out, _ = mask_space.transform(out, space_below.axes)

        return out, meta

    def __repr__(self):
        return "{}(R={})".format(
            self.__class__.__name__,
            self.dropout_rate)

def _softmax(smax, ratio):
    smax.W = smax.W * (1-ratio)
    return smax

def _sentence_convolution(conv_layer, ratio):
    conv_layer.W = conv_layer.W * (1-ratio)
    return conv_layer

def _identity(layer, ratio):
    return layer

__function_mapping = {
    'Softmax' : _softmax,
    'SentenceConvolution' : _sentence_convolution,
    'Tanh' : _identity,
    'Bias' : _identity,
    'MaxFolding' : _identity,
    'SumFolding' : _identity,
    'WordEmbedding' : _identity,
    'DictionaryEncoding' : _identity
    }

def remove_dropout(model):

    new_model = []
    ratio = 0
    for layer in model.layers:
        if layer.__class__.__name__=='Dropout':
            ratio = layer.dropout_rate
        else:
            if ratio==0:
                new_model.append(layer)
            else:
                new_model.append(__function_mapping[layer.__class__.__name__](layer, ratio))
                ratio=0

    return CSM(layers=new_model)
