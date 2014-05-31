__author__ = 'albandemiraj'

from cpu.space import CPUSpace
import numpy as np
import cpu.model.layer
from cpu.model.model import CSM
from cpu.model.transfer import SentenceConvolution
from cpu.model.transfer import Softmax
from cpu.model.transfer import Linear

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
    new_smax = Softmax(n_classes=smax.n_classes,
                       n_input_dimensions=smax.n_input_dimensions)

    new_smax.W = smax.W.copy() * (1-ratio)
    new_smax.b = smax.b.copy()
    return new_smax

def _linear(linear, ratio):
    new_linear = Linear(n_input=linear.n_input, n_output=linear.n_output)

    new_linear.W = linear.W * (1-ratio)
    return new_linear

def _sentence_convolution(conv_layer, ratio):
    new_conv = SentenceConvolution(
                n_feature_maps=conv_layer.n_feature_maps,
                kernel_width=conv_layer.kernel_width,
                n_channels=conv_layer.n_channels,
                n_input_dimensions=conv_layer.n_input_dimensions)

    new_conv.W = conv_layer.W.copy() * (1-ratio)
    return new_conv

def _identity(layer, ratio):
    return layer

__function_mapping = {
    'Softmax' : _softmax,
    'SentenceConvolution' : _sentence_convolution,
    'Linear': _linear,
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
