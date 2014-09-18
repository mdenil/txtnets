__author__ = 'mdenil'


import cpu.model.cost
import cpu.model.embedding
import cpu.model.encoding
import cpu.model.model
import cpu.model.nonlinearity
import cpu.model.pooling
import cpu.model.transfer
import cpu.model.repeat_layer
import cpu.model.transport
import cpu.model.dropout

import gpu.model.cost
import gpu.model.embedding
import gpu.model.encoding
import gpu.model.model
import gpu.model.nonlinearity
import gpu.model.pooling
import gpu.model.transfer
import gpu.model.repeat_layer
import gpu.model.transport
import gpu.model.dropout


__host_device_component_mapping = {
    cpu.model.cost.CrossEntropy: gpu.model.cost.CrossEntropy,

    cpu.model.embedding.WordEmbedding: gpu.model.embedding.WordEmbedding,

    cpu.model.encoding.DictionaryEncoding: gpu.model.encoding.DictionaryEncoding,

    cpu.model.model.CSM: gpu.model.model.CSM,
    cpu.model.model.TaggedModelCollection: gpu.model.model.TaggedModelCollection,

    cpu.model.nonlinearity.Relu: gpu.model.nonlinearity.Relu,
    cpu.model.nonlinearity.Tanh: gpu.model.nonlinearity.Tanh,

    cpu.model.pooling.KMaxPooling: gpu.model.pooling.KMaxPooling,
    cpu.model.pooling.MaxFolding: gpu.model.pooling.MaxFolding,
    cpu.model.pooling.SumFolding: gpu.model.pooling.SumFolding,

    cpu.model.transfer.Bias: gpu.model.transfer.Bias,
    cpu.model.transfer.Linear: gpu.model.transfer.Linear,
    cpu.model.transfer.SentenceConvolution: gpu.model.transfer.SentenceConvolution,
    cpu.model.transfer.Softmax: gpu.model.transfer.Softmax,
    cpu.model.transfer.AxisReduction: gpu.model.transfer.AxisReduction,
    cpu.model.transfer.ReshapeForDocuments: gpu.model.transfer.ReshapeForDocuments,

    cpu.model.repeat_layer.RepeatLayer: gpu.model.repeat_layer.RepeatLayer,

    cpu.model.transport.HostToDevice: gpu.model.transport.HostToDevice,
    cpu.model.transport.DeviceToHost: gpu.model.transport.DeviceToHost,

    cpu.model.dropout.Dropout: gpu.model.dropout.Dropout,
}

__device_host_component_mapping = dict(
    (d, h) for h, d in __host_device_component_mapping.iteritems()
)


def get_cpu_analog(gpu_class):
    return __device_host_component_mapping[gpu_class]


########


def move_to_gpu(model):
    gpu_layers = [_move_layer_to_gpu(layer) for layer in model.layers]
    return gpu.model.model.CSM(layers=gpu_layers)


def _move_embedding(embedding):
    return gpu.model.embedding.WordEmbedding(
        dimension=embedding.dimension,
        vocabulary_size=embedding.vocabulary_size,
        padding=embedding.padding,
        E=embedding.E)


def _move_encoding(encoding):
    return gpu.model.encoding.DictionaryEncoding(vocabulary=encoding.vocabulary)


def _move_relu(relu):
    return gpu.model.nonlinearity.Relu()


def _move_tanh(tanh):
    return gpu.model.nonlinearity.Tanh()


def _move_kmax_pooling(kmax):
    return gpu.model.pooling.KMaxPooling(k=kmax.k, k_dynamic=kmax.k_dynamic)


def _move_bias(bias):
    return gpu.model.transfer.Bias(
        n_input_dims=bias.n_input_dims,
        n_feature_maps=bias.n_feature_maps,
        b=bias.b)


def _move_linear(linear):
    return gpu.model.transfer.Linear(
        n_input=linear.n_input,
        n_output=linear.n_output,
        W=linear.W)


def _move_sentence_convolution(sconv):
    return gpu.model.transfer.SentenceConvolution(
        n_feature_maps=sconv.n_feature_maps,
        kernel_width=sconv.kernel_width,
        n_input_dimensions=sconv.n_input_dimensions,
        n_channels=sconv.n_channels,
        W=sconv.W)


def _move_softmax(softmax):
    return gpu.model.transfer.Softmax(
        n_classes=softmax.n_classes,
        n_input_dimensions=softmax.n_input_dimensions,
        W=softmax.W,
        b=softmax.b)


def _move_axis_reduction(axis_reduction):
    return gpu.model.transfer.AxisReduction(axis=axis_reduction.axis)


def _move_reshape_for_documents(rfd):
    return gpu.model.transfer.ReshapeForDocuments()


def _move_dropout(dropout):
    return gpu.model.dropout.Dropout(
        axes=dropout.axes,
        dropout_rate=dropout.dropout_rate)


__host_to_device_move_functions = {
    cpu.model.embedding.WordEmbedding: _move_embedding,

    cpu.model.encoding.DictionaryEncoding: _move_encoding,

    cpu.model.nonlinearity.Relu: _move_relu,
    cpu.model.nonlinearity.Tanh: _move_tanh,

    cpu.model.pooling.KMaxPooling: _move_kmax_pooling,

    cpu.model.transfer.Bias: _move_bias,
    cpu.model.transfer.Linear: _move_linear,
    cpu.model.transfer.SentenceConvolution: _move_sentence_convolution,
    cpu.model.transfer.Softmax: _move_softmax,
    cpu.model.transfer.AxisReduction: _move_axis_reduction,
    cpu.model.transfer.ReshapeForDocuments: _move_reshape_for_documents,

    cpu.model.dropout.Dropout: _move_dropout,
}


def _move_layer_to_gpu(layer):
    return __host_to_device_move_functions[layer.__class__](layer)