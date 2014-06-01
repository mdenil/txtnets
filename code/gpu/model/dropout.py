__author__ = 'mdenil'

import numpy as np

import gpu.space
import gpu.model.layer
import generic.model.dropout

from gpu.model.model import CSM
from gpu.model.transfer import SentenceConvolution
from gpu.model.transfer import Softmax
from gpu.model.transfer import Linear

import pycuda.autoinit
import pycuda.curandom

_dropout_mask_kernel = pycuda.elementwise.ElementwiseKernel(
    "float *x, float q",  # q is the keep probability (i.e. 1-p)
    "x[i] = (float)(x[i] < q)",
    "dropout_mask")


class Dropout(generic.model.dropout.Dropout, gpu.model.layer.Layer):
    def __init__(self, *args, **kwargs):
        super(Dropout, self).__init__(*args, **kwargs)
        self.__acquire_device_kernels()

    def _get_mask(self, shape):
        mask = pycuda.gpuarray.empty(shape=shape, dtype=np.float32)
        self._rng.fill_uniform(mask)
        _dropout_mask_kernel(mask, np.float32(1.0 - self.dropout_rate))

        mask_space = gpu.space.GPUSpace.infer(mask, self.axes)

        return mask, mask_space

    def __acquire_device_kernels(self):
        self._rng = pycuda.curandom.MRG32k3aRandomNumberGenerator()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_rng']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__acquire_device_kernels()

    def move_to_cpu(self):
        from gpu.model.host_device_component_mapping import get_cpu_analog
        cpu_class = get_cpu_analog(self.__class__)

        return cpu_class(
            axes=self.axes,
            dropout_rate=self.dropout_rate)

def _remove_dropout_softmax(smax, ratio):
    new_smax = Softmax(
        n_classes=smax.n_classes,
        n_input_dimensions=smax.n_input_dimensions)

    new_smax.W = smax.W.copy() * (1-ratio)
    new_smax.b = smax.b.copy()
    return new_smax


def _remove_dropout_linear(linear, ratio):
    new_linear = Linear(
        n_input=linear.n_input,
        n_output=linear.n_output)

    new_linear.W = linear.W.copy() * (1-ratio)
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
    'Softmax': _remove_dropout_softmax,
    'SentenceConvolution': _sentence_convolution,
    'Linear': _remove_dropout_linear,
    'Tanh': _identity,
    'Bias': _identity,
    'MaxFolding': _identity,
    'SumFolding': _identity,
    'WordEmbedding': _identity,
    'DictionaryEncoding': _identity
}


def remove_dropout(model):
    new_model = []
    ratio = 0
    for layer in model.layers:
        if layer.__class__.__name__ == 'Dropout':
            ratio = layer.dropout_rate
        else:
            if ratio == 0:
                new_model.append(layer)
            else:
                new_model.append(__function_mapping[layer.__class__.__name__](layer, ratio))
                ratio = 0

    return CSM(layers=new_model)
