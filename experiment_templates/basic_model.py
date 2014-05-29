__author__ = 'mdenil'

from cpu.model.model import CSM
from cpu.model.encoding import DictionaryEncoding
from cpu.model.embedding import WordEmbedding
from cpu.model.transfer import SentenceConvolution
from cpu.model.transfer import Bias
from cpu.model.pooling import KMaxPooling
from cpu.model.transfer import ReshapeForDocuments
from cpu.model.nonlinearity import Tanh
from cpu.model.nonlinearity import Relu
from cpu.model.transfer import Softmax
from cpu.model.dropout import Dropout


def get_model(encoding):

    return CSM(
        layers=[
            DictionaryEncoding(vocabulary=encoding),

            WordEmbedding(
                dimension={{embedding_dimension}},
                vocabulary_size=len(encoding),
                padding=encoding['PADDING']),

            {% for layer in word_layers %}
            {% set layer_index = loop.index0 %}

            SentenceConvolution(
                n_feature_maps={{layer.n_feature_maps}},
                kernel_width={{layer.kernel_width}},
                n_channels={{layer.n_channels}},
                n_input_dimensions=1),

            Bias(
                n_input_dims=1,
                n_feature_maps={{layer.n_feature_maps}}),

            KMaxPooling(k={{layer.k_pooling}}, k_dynamic={{layer.k_dynamic}} if {{layer.k_dynamic}} > 0 else None),

            {{layer.nonlinearity}}(),

            {% endfor %}

            ReshapeForDocuments(),

            {% for layer in sentence_layers %}
            {% set layer_index = loop.index0 %}

            SentenceConvolution(
                n_feature_maps={{layer.n_feature_maps}},
                kernel_width={{layer.kernel_width}},
                n_channels={{layer.n_channels}},
                n_input_dimensions=1),

            Bias(
                n_input_dims=1,
                n_feature_maps={{layer.n_feature_maps}}),

            KMaxPooling(k={{layer.k_pooling}}, k_dynamic={{layer.k_dynamic}} if {{layer.k_dynamic}} > 0 else None),

            {{layer.nonlinearity}}(),

            {% endfor %}

            {% if dropout %}
            Dropout(('b', 'd', 'f', 'w'), 0.5),
            {% endif %}

            Softmax(
                n_classes={{n_classes}},
                n_input_dimensions={{softmax_input_dimensions}}),
            ])