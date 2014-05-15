__author__ = 'mdenil'


import numpy as np
import os
import time
import random
import simplejson as json
import cPickle as pickle

from gpu.model.model import CSM
from gpu.model.encoding import DictionaryEncoding
from gpu.model.embedding import WordEmbedding
# from gpu.model.transfer import SentenceConvolution
# from gpu.model.transfer import Bias
# from gpu.model.pooling import SumFolding
# from gpu.model.pooling import MaxFolding
# from gpu.model.pooling import KMaxPooling
# from gpu.model.nonlinearity import Tanh
# from gpu.model.transfer import Softmax
from gpu.model.transfer import AxisReduction
from gpu.model.model import TaggedModelCollection

from gpu.optimize.objective import ContrastiveMultilingualEmbeddingObjective

from gpu.optimize.sgd import SGD
# from gpu.optimize.regularizer import L2Regularizer
from gpu.optimize.update_rule import AdaGrad
from gpu.optimize.data_provider import PaddedParallelSequenceMinibatchProvider
from generic.optimize.data_provider import TaggedProviderCollection


def replace_unknowns(data, dictionary, unknown):
    new_data = []
    for sentence in data:
        new_data.append(
            [word if word in dictionary else unknown for word in sentence]
        )
    return new_data


def run():
    # random.seed(435)
    # np.random.seed(2342)
    np.set_printoptions(linewidth=100)

    data_dir = os.path.join("../data", "europarlv7")

    with open(os.path.join(data_dir, "europarl-v7.de-en.en.tokens.clean.json")) as data_file:
        english_data = json.load(data_file)

    with open(os.path.join(data_dir, "europarl-v7.de-en.en.tokens.clean.dictionary.encoding.json")) as dictionary_file:
        english_dictionary = json.load(dictionary_file)

    with open(os.path.join(data_dir, "europarl-v7.de-en.de.tokens.clean.json")) as data_file:
        german_data = json.load(data_file)

    with open(os.path.join(data_dir, "europarl-v7.de-en.de.tokens.clean.dictionary.encoding.json")) as dictionary_file:
        german_dictionary = json.load(dictionary_file)

    # english_data = english_data[:10000]
    # german_data = german_data[:10000]

    english_data = replace_unknowns(english_data, english_dictionary, 'UNKNOWN')
    german_data = replace_unknowns(german_data, german_dictionary, 'UNKNOWN')

    batch_size = 100

    assert len(english_data) == len(german_data)
    print len(english_data) / batch_size

    parallel_en_de_provider = PaddedParallelSequenceMinibatchProvider(
        X1=list(english_data),
        X2=list(german_data),
        batch_size=batch_size,
        padding='PADDING',
    )

    multilingual_parallel_provider = TaggedProviderCollection({
        ('en', 'de'): parallel_en_de_provider
    })

    english_model = CSM(
        layers=[
            DictionaryEncoding(vocabulary=english_dictionary),

            WordEmbedding(
                dimension=40,
                vocabulary_size=len(english_dictionary)),

            AxisReduction(axis='w'),

            # SentenceConvolution(
            #     n_feature_maps=15,
            #     kernel_width=10,
            #     n_channels=1,
            #     n_input_dimensions=12),
            #
            # SumFolding(),
            #
            # KMaxPooling(k=17),
            #
            # Bias(
            #     n_input_dims=6,
            #     n_feature_maps=15),
            #
            # Tanh(),
            ]
        )

    german_model = CSM(
        layers=[
            DictionaryEncoding(vocabulary=german_dictionary),

            WordEmbedding(
                dimension=40,
                vocabulary_size=len(german_dictionary)),

            AxisReduction(axis='w'),

            # SentenceConvolution(
            #     n_feature_maps=15,
            #     kernel_width=10,
            #     n_channels=1,
            #     n_input_dimensions=12),
            #
            # SumFolding(),
            #
            # KMaxPooling(k=17),
            #
            # Bias(
            #     n_input_dims=6,
            #     n_feature_maps=15),
            #
            # Tanh(),

            ]
        )

    print english_model
    print german_model

    model = TaggedModelCollection({
        'en': english_model,
        'de': german_model,
    })

    # regularizer = L2Regularizer(lamb=1e-4)

    objective = ContrastiveMultilingualEmbeddingObjective(
        tagged_parallel_sequence_provider=multilingual_parallel_provider,
        n_contrastive_samples=10,
        margin=40.0)

    # objective = CostMinimizationObjective(
    #     cost=cost_function,
    #     data_provider=train_data_provider,
    #     regularizer=regularizer)

    update_rule = AdaGrad(
        # gamma=0.01,
        gamma=0.1,
        model_template=model)

    optimizer = SGD(
        model=model,
        objective=objective,
        update_rule=update_rule)

    time_start = time.time()

    costs = []
    for batch_index, iteration_info in enumerate(optimizer):
        costs.append(iteration_info['cost'])

        # print costs[-1]

        if batch_index % 10 == 0:
            print "B: {}, E: {}, C: {}, Param size: {}".format(
                batch_index,
                # This epoch count will be inaccurate when I move to multilingual
                (batch_index // parallel_en_de_provider.batches_per_epoch) + 1,
                costs[-1],
                np.mean(np.abs(model.pack())))

        if batch_index % 100 == 0:
            with open("model.pkl", 'w') as model_file:
                pickle.dump(model.move_to_cpu(), model_file, protocol=-1)

        # if batch_index % 1000 == 0 and batch_index > 0:
        #     with open("model_optimization.pkl", 'w') as model_file:
        #         pickle.dump(optimizer, model_file, protocol=-1)

        # if batch_index == 500:
        #     break

    time_end = time.time()

    print "Time elapsed: {}s".format(time_end - time_start)


if __name__ == "__main__":
    run()